#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from cuda.core.experimental import Device
from torch.distributed.tensor import DTensor, distribute_tensor

import nvshmem.core as nvshmem

THIS_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import util as bench_util


def _get_partitioning(partition: str, replication_factor: int):
    if partition == "row":
        return bench_util.row_partitioning(replication_factor=replication_factor)
    if partition == "column":
        return bench_util.column_partitioning(replication_factor=replication_factor)
    if partition == "block":
        return bench_util.two_dimensional_partitioning(replication_factor=replication_factor)
    raise ValueError(f"Unsupported partitioning: {partition}")


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


def _sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()


@dataclass
class BenchResult:
    method: str
    m: int
    n: int
    k: int
    a_partition: str
    b_partition: str
    c_partition: str
    replication_factor: int
    warmup_iters: int
    measure_iters: int
    world_size: int
    median_seconds: float
    min_seconds: float
    max_seconds: float
    median_tflops: float
    max_tflops: float
    flops: float
    theoretical_tflops: float | None


def _execute_one_iteration(
    *,
    method: str,
    dt_a: DTensor,
    dt_b: DTensor,
    dt_c: DTensor | None,
    nvshmem_stream: nvshmem.nvshmem_types.NvshmemStream | None,
) -> None:
    if method == "dtensor":
        dt_out = torch.matmul(dt_a, dt_b)
        requires_redistribute, new_placements = bench_util.materialized_placements(
            dt_out.device_mesh, dt_out.placements
        )
        if requires_redistribute:
            dt_out = dt_out.redistribute(dt_out.device_mesh, new_placements)
        _ = dt_out.to_local()
        _sync_device()
        return

    # Lazy import to avoid pulling these dependencies when benchmarking plain DTensor.
    import dtensor_utils as dt

    assert dt_c is not None
    dt_c.to_local().zero_()

    if method == "stationary_c":
        dt.execute_stationary_c(dt_a, dt_b, dt_c, local_only=True)
    elif method == "stationary_b":
        dt.execute_stationary_b(dt_a, dt_b, dt_c, local_only=True)
    else:
        raise ValueError(f"Unsupported method: {method}")

    if nvshmem_stream is not None:
        nvshmem.barrier_all(stream=nvshmem_stream)
    _sync_device()


def run_worker(args: argparse.Namespace) -> None:
    if args.method not in {"dtensor", "stationary_c", "stationary_b"}:
        raise ValueError(f"Unsupported method: {args.method}")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise RuntimeError("No CUDA devices are available.")
    gpu_id = rank % device_count
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    use_stationary = args.method in {"stationary_c", "stationary_b"}
    if use_stationary and os.environ.get("TORCH_DTENSOR_USE_NVSHMEM", "0") != "1":
        raise RuntimeError(
            "stationary methods require TORCH_DTENSOR_USE_NVSHMEM=1 "
            "in the worker process environment"
        )

    nvshmem_stream = _init_nvshmem(rank, world_size, gpu_id) if use_stationary else None

    dt_a = None
    dt_b = None
    dt_c = None
    full_a = None
    full_b = None
    full_c = None

    try:
        a_p = _get_partitioning(args.a_partition, args.replication_factor)
        b_p = _get_partitioning(args.b_partition, args.replication_factor)
        c_p = _get_partitioning(args.c_partition, args.replication_factor)

        full_a = torch.randn(args.m, args.k, dtype=torch.float32, device=device)
        full_b = torch.randn(args.k, args.n, dtype=torch.float32, device=device)
        full_c = torch.zeros(args.m, args.n, dtype=torch.float32, device=device)

        dt_a = distribute_tensor(full_a, *a_p)
        dt_b = distribute_tensor(full_b, *b_p)
        if use_stationary:
            dt_c = distribute_tensor(full_c, *c_p)

            import dtensor_utils as dt

            a_tile_numel = dt.tile_shape(dt_a)[0] * dt.tile_shape(dt_a)[1]
            b_tile_numel = dt.tile_shape(dt_b)[0] * dt.tile_shape(dt_b)[1]
            # Conservative but bounded scratch size for get_tile staging.
            scratch_elements = 4 * max(a_tile_numel, b_tile_numel)
            dt.init_scratch(scratch_elements, dt_a.dtype)

        dist.barrier()

        for _ in range(args.warmup_iters):
            _execute_one_iteration(
                method=args.method,
                dt_a=dt_a,
                dt_b=dt_b,
                dt_c=dt_c,
                nvshmem_stream=nvshmem_stream,
            )
            dist.barrier()

        durations: list[float] = []
        for _ in range(args.measure_iters):
            dist.barrier()
            t0 = time.perf_counter()
            _execute_one_iteration(
                method=args.method,
                dt_a=dt_a,
                dt_b=dt_b,
                dt_c=dt_c,
                nvshmem_stream=nvshmem_stream,
            )
            dist.barrier()
            t1 = time.perf_counter()

            local_elapsed = torch.tensor(
                [t1 - t0], dtype=torch.float64, device=device
            )
            dist.all_reduce(local_elapsed, op=dist.ReduceOp.MAX)
            if rank == 0:
                durations.append(float(local_elapsed.item()))

        if rank == 0:
            flops = float(2.0 * args.m * args.n * args.k)
            tflops = [flops / d / 1e12 for d in durations]
            result = BenchResult(
                method=args.method,
                m=args.m,
                n=args.n,
                k=args.k,
                a_partition=args.a_partition,
                b_partition=args.b_partition,
                c_partition=args.c_partition,
                replication_factor=args.replication_factor,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                world_size=world_size,
                median_seconds=statistics.median(durations),
                min_seconds=min(durations),
                max_seconds=max(durations),
                median_tflops=statistics.median(tflops),
                max_tflops=max(tflops),
                flops=flops,
                theoretical_tflops=args.theoretical_tflops,
            )
            os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
            with open(args.result_file, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, indent=2, sort_keys=True)

    finally:
        try:
            dist.barrier()
        except Exception:
            pass

        if use_stationary:
            try:
                import dtensor_utils as dt

                if dt_a is not None:
                    nvshmem.free_tensor(dt_a.nvshmem_base())
                if dt_b is not None:
                    nvshmem.free_tensor(dt_b.nvshmem_base())
                if dt_c is not None:
                    nvshmem.free_tensor(dt_c.nvshmem_base())
                dt.free_get_tile_scratch()
            except Exception:
                pass

            try:
                if nvshmem_stream is not None:
                    nvshmem.barrier_all(stream=nvshmem_stream)
                nvshmem.finalize()
            except Exception:
                pass

        try:
            dist.destroy_process_group()
        except Exception:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-case DTensor/Universal GEMM benchmark")
    parser.add_argument("--method", choices=["dtensor", "stationary_c", "stationary_b"], required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--a-partition", choices=["row", "column", "block"], required=True)
    parser.add_argument("--b-partition", choices=["row", "column", "block"], required=True)
    parser.add_argument("--c-partition", choices=["row", "column", "block"], required=True)
    parser.add_argument("--replication-factor", type=int, required=True)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--theoretical-tflops", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_worker(args)


if __name__ == "__main__":
    main()
