#!/usr/bin/env python3
# example.py
import argparse
import os
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor

import benchmark.util
import dtensor_utils as dt
import dtensor_mm_handler
import nvshmem.core as nvshmem
from cuda.core.experimental import Device

def _init_nvshmem(rank: int, world_size: int, gpu_id: int) -> nvshmem.nvshmem_types.NvshmemStream:
    local_rank = int(os.environ.get("LOCAL_RANK", gpu_id))
    dev = Device(local_rank)
    dev.set_current()

    uid = nvshmem.get_unique_id(empty=(local_rank != 0))
    uid_bytes = uid._data.view(np.uint8).copy()
    uid_tensor = torch.from_numpy(uid_bytes).cuda()
    dist.broadcast(uid_tensor, src=0)
    dist.barrier()
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)
    nvshmem.init(
        device=dev,
        uid=uid,
        rank=local_rank,
        nranks=world_size,
        initializer_method="uid",
    )
    return nvshmem.NvshmemStream(torch.cuda.current_stream())

def get_partitioning(partition: str, replication_factor: int):
    if partition == 'row':
        return benchmark.util.row_partitioning(replication_factor=replication_factor)
    elif partition == 'column':
        return benchmark.util.column_partitioning(replication_factor=replication_factor)
    elif partition == 'block':
        return benchmark.util.two_dimensional_partitioning(replication_factor=replication_factor)
    else:
        raise RuntimeError(f'error: unsupported partitioning {partition}')

def run_matmul_benchmark(
    m: int,
    n: int,
    k: int,
    a_partition: str,
    b_partition: str,
    c_partition: str,
    a_replication_factor: int,
    b_replication_factor: int,
    c_replication_factor: int,
    stationary_method: str,
):
    # Init pytorch.dist
    dist.init_process_group(backend='nccl')

    assert os.environ.get("TORCH_DTENSOR_USE_NVSHMEM", "0") == "1", (
        "Expected TORCH_DTENSOR_USE_NVSHMEM=1 for this example."
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus
    torch.cuda.set_device(gpu_id)
    device = torch.cuda.current_device()

    print(f'Rank {rank}/{world_size} running on GPU {gpu_id}/{num_gpus}')

    stream = _init_nvshmem(rank, world_size, gpu_id)

    dtensor_mm_handler.enable(stationary_method=stationary_method)

    a_p = get_partitioning(a_partition, a_replication_factor)
    b_p = get_partitioning(b_partition, b_replication_factor)
    c_p = get_partitioning(c_partition, c_replication_factor)

    global_a = torch.randn(m, k, dtype=torch.float32, device=device)*50 + 50
    global_b = torch.randn(k, n, dtype=torch.float32, device=device)*50 + 50
    global_c = torch.zeros(m, n, dtype=torch.float32, device=device)*50 + 50

    dt_a = distribute_tensor(global_a, *a_p)
    dt_b = distribute_tensor(global_b, *b_p)
    dt_c = distribute_tensor(global_c, *c_p)

    dt.init_scratch(2 * k * n, dt_b.dtype)

    if rank == 0:
        print(f'Multiply A {dt_a.shape} by B {dt_b.shape} -> C {dt_c.shape}')
        print(f'Tile grids are A {dt.grid_shape(dt_a)}, B {dt.grid_shape(dt_b)}, and C {dt.grid_shape(dt_c)}')

    n_iterations = 10

    durations = []

    for i in range(n_iterations):
        dist.barrier()
        begin = time.time()
        torch.addmm(dt_c, dt_a, dt_b, out=dt_c)
        nvshmem.barrier_all(stream=stream)
        torch.cuda.current_stream().synchronize()
        end = time.time()
        duration = end - begin

        durations.append(duration)

    if rank == 0:
        dtensor_mm_handler.print_stats()

    ref_durations = []
    for i in range(n_iterations):
        begin = time.time()
        torch.addmm(global_c, global_a, global_b, out=global_c)
        torch.cuda.current_stream().synchronize()
        end = time.time()
        duration = end - begin

        ref_durations.append(duration)

    full_c = dt_c.full_tensor()
    if rank == 0:
        torch.testing.assert_close(
            full_c.cpu(),
            global_c.cpu(),
            rtol=1e-4,
            atol=1e-5,
        )

    if rank == 0:
        gflops = 1e-9 * ((2 * m * n * k) + (3 * m * n))

        all_gflops = [gflops / duration for duration in durations]
        reference_gflops = [gflops / duration for duration in ref_durations]
        print(f"Max Distributed GFLOPs: {np.max(all_gflops)}")
        print(f"Median Distributed GFLOPs: {np.median(all_gflops)}")
        print(f"Max Reference (Single GPU) GFLOPs: {np.max(reference_gflops)}")
        print(f"Median Reference (Single GPU) GFLOPs: {np.median(reference_gflops)}")

        print(f"Median speedup is {np.median(all_gflops) / np.median(reference_gflops)} over a single GPU.")

    # Collective, ordered teardown to avoid finalizing NVSHMEM while peers still hold/use buffers.
    dist.barrier()
    nvshmem.barrier_all(stream=stream)
    torch.cuda.current_stream().synchronize()

    nvshmem.free_tensor(dt_a.nvshmem_base())
    nvshmem.free_tensor(dt_b.nvshmem_base())
    nvshmem.free_tensor(dt_c.nvshmem_base())
    dt.free_get_tile_scratch()
    dtensor_mm_handler.disable()

    # Ensure all ranks complete frees before NVSHMEM finalize.
    dist.barrier()
    nvshmem.barrier_all(stream=stream)
    torch.cuda.current_stream().synchronize()

    del full_c, dt_a, dt_b, dt_c
    nvshmem.finalize()
    dist.destroy_process_group()

def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed matrix multiplication example")
    parser.add_argument("--m", type=int, default=8 * 1024, help="Number of rows in matrix A")
    parser.add_argument("--n", type=int, default=8 * 1024, help="Number of columns in matrix B")
    parser.add_argument("--k", type=int, default=8 * 1024, help="Number of columns in A / rows in B")
    parser.add_argument(
        "--a-partition",
        type=str,
        choices=["row", "column", "block"],
        default="row",
        help="Partitioning strategy for matrix A",
    )
    parser.add_argument(
        "--b-partition",
        type=str,
        choices=["row", "column", "block"],
        default="row",
        help="Partitioning strategy for matrix B",
    )
    parser.add_argument(
        "--c-partition",
        type=str,
        choices=["row", "column", "block"],
        default="row",
        help="Partitioning strategy for matrix C",
    )
    parser.add_argument(
        "--replication",
        type=int,
        default=None,
        help="Replication factor for A, B, and C (default: 1; mutually exclusive with per-matrix replication flags)",
    )
    parser.add_argument(
        "--a-replication",
        type=int,
        default=None,
        help="Replication factor for A (default: 1)",
    )
    parser.add_argument(
        "--b-replication",
        type=int,
        default=None,
        help="Replication factor for B (default: 1)",
    )
    parser.add_argument(
        "--c-replication",
        type=int,
        default=None,
        help="Replication factor for C (default: 1)",
    )
    parser.add_argument(
        "--stationary-method",
        type=str,
        choices=["auto", "stationary_c", "stationary_b"],
        default="auto",
        help="Select stationary execution method used by addmm handler",
    )
    args = parser.parse_args()

    has_any_matrix_replication = (
        args.a_replication is not None
        or args.b_replication is not None
        or args.c_replication is not None
    )
    if args.replication is not None and has_any_matrix_replication:
        parser.error(
            "Cannot pass --replication together with --a-replication/--b-replication/--c-replication."
        )

    if args.replication is not None:
        if args.replication <= 0:
            parser.error("--replication must be >= 1.")
        a_replication = args.replication
        b_replication = args.replication
        c_replication = args.replication
    else:
        if (
            (args.a_replication is not None and args.a_replication <= 0)
            or (args.b_replication is not None and args.b_replication <= 0)
            or (args.c_replication is not None and args.c_replication <= 0)
        ):
            parser.error("--a-replication/--b-replication/--c-replication must be >= 1.")
        a_replication = 1 if args.a_replication is None else args.a_replication
        b_replication = 1 if args.b_replication is None else args.b_replication
        c_replication = 1 if args.c_replication is None else args.c_replication

    run_matmul_benchmark(
        args.m,
        args.n,
        args.k,
        args.a_partition,
        args.b_partition,
        args.c_partition,
        a_replication,
        b_replication,
        c_replication,
        args.stationary_method,
    )


if __name__ == "__main__":
    main()
