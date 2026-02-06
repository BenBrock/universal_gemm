#!/usr/bin/env python3
# example.py
import os
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

import benchmark.util
import dtensor_utils as dt
import dtensor_mm_handler
import nvshmem.core as nvshmem
from cuda.core.experimental import Device

def main():
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

    group = dist.group.WORLD

    # Initialize NVSHMEM4Py using a UID broadcasted via torch.distributed
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
    stream = nvshmem.NvshmemStream(torch.cuda.current_stream())

    dtensor_mm_handler.enable()

    m = 8*1024
    n = 8*1024
    k = 8*1024

    '''
    NOTE: I *think* things should work for uneven tile sizes?
    assert m % world_size == 0, "m must be divisible by world_size for even row sharding"
    assert k % world_size == 0, "k must be divisible by world_size for even row sharding"
    '''

    a_p = benchmark.util.row_partitioning()
    b_p = benchmark.util.row_partitioning()
    c_p = benchmark.util.row_partitioning()

    global_a = torch.randn(m, k, dtype=torch.float32, device=device)*50 + 50
    global_b = torch.randn(k, n, dtype=torch.float32, device=device)*50 + 50
    global_c = torch.zeros(m, n, dtype=torch.float32, device=device)*50 + 50

    dt_a = distribute_tensor(global_a, *a_p)
    dt_b = distribute_tensor(global_b, *b_p)
    dt_c = distribute_tensor(global_c, *c_p)

    b_tile_shape = dt.tile_shape(dt_b)
    b_tile_numel = b_tile_shape[0] * b_tile_shape[1]
    b_tiles_per_round = dt.grid_shape(dt_b)[0]
    scratch_elements = b_tile_numel * b_tiles_per_round
    dt.init_scratch(scratch_elements, dt_b.dtype)

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

    nvshmem.free_tensor(dt_a.nvshmem_base())
    nvshmem.free_tensor(dt_b.nvshmem_base())
    nvshmem.free_tensor(dt_c.nvshmem_base())
    dt.free_get_tile_scratch()
    nvshmem.finalize()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
