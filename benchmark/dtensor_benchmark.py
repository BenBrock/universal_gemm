import os
import sys
import time
import util
import numpy as np
import torch
import torch.distributed as dist
import argparse
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard, Partial

from util import row_partitioning,column_partitioning,two_dimensional_partitioning

# Data type for computation
DTYPE = torch.float32

# Device type
DEVICE_TYPE = "cuda"
# DEVICE_TYPE = "xpu"

# PyTorch Dist Communication backend
DIST_BACKEND = "nccl"
# DIST_BACKEND = "xccl"

# Whether to check for correctness
VERIFY_RESULT = False

# Disable TF32
if DEVICE_TYPE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def run_matmul_benchmark(m: int, n: int, k: int, a_partition: str, b_partition: str, replication_factor: int):
    # Initialize the distributed environment
    dist.init_process_group(backend=DIST_BACKEND)

    if DIST_BACKEND != "nccl" and DIST_BACKEND != "xccl":
        raise RuntimeError(f"PyTorch Dist backend {DIST_BACKEND} not supported.")

    if DEVICE_TYPE != "cuda" and DEVICE_TYPE != "xpu":
        raise RuntimeError(f"PyTorch device type {DEVICE_TYPE} not supported.")

    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Get the number of GPUs available on this node
    if DEVICE_TYPE == "cuda":
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available!")
    elif DEVICE_TYPE == "xpu":
        num_gpus = torch.xpu.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available!")
    elif DEVICE_TYPE == "cpu":
        pass
    else:
        raise RuntimeError(f"ERROR!")

    # Assign GPU based on rank modulo number of GPUs
    if DEVICE_TYPE == "cuda":
        gpu_id = rank % num_gpus
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    elif DEVICE_TYPE == "xpu":
        gpu_id = rank % num_gpus
        torch.xpu.set_device(gpu_id)
        device = torch.device(f"xpu:{gpu_id}")
    elif DEVICE_TYPE == "cpu":
        device = torch.device("cpu")
        pass
    else:
        raise RuntimeError(f"ERROR!")

    print(f"Hello, world! I am process {rank} / {world_size}, using GPU {gpu_id} / {num_gpus}")

    if rank == 0:
        print(f"Multiplying matrices A {m} x {k}, B {k} x {n} -> C {m} x {n}")

    # Create global tensors first with specified dtype
    # Scale randn to [0, 100] by multiplying by 50 (to get [-50, 50] roughly) and adding 50
    global_A = (torch.randn(m, k, dtype=DTYPE, device=device) * 50 + 50)
    global_B = (torch.randn(k, n, dtype=DTYPE, device=device) * 50 + 50)

    # Set up partitioning based on command line arguments
    if a_partition == "row":
        a_partitioning = row_partitioning(replication_factor=replication_factor)
    elif a_partition == "column":
        a_partitioning = column_partitioning(replication_factor=replication_factor)
    elif a_partition == "block":
        a_partitioning = two_dimensional_partitioning(replication_factor=replication_factor)
    else:
        raise ValueError(f"Unknown partitioning strategy for A: {a_partition}")

    if b_partition == "row":
        b_partitioning = row_partitioning(replication_factor=replication_factor)
    elif b_partition == "column":
        b_partitioning = column_partitioning(replication_factor=replication_factor)
    elif b_partition == "block":
        b_partitioning = two_dimensional_partitioning(replication_factor=replication_factor)
    else:
        raise ValueError(f"Unknown partitioning strategy for B: {b_partition}")

    if rank == 0:
        print(f"a: {a_partitioning}, b: {b_partitioning}")

    if rank == 0:
        print(f"Distributing tensors...")

    dt_A = distribute_tensor(global_A, *a_partitioning)
    dt_B = distribute_tensor(global_B, *b_partitioning)

    if rank == 0:
        print(f"Distributed tensors!")
        print(f"Multiplying {dt_A} by {dt_B}")

    if VERIFY_RESULT:
        # Perform local matrix multiplication for verification
        global_A = dt_A.full_tensor()
        global_B = dt_B.full_tensor()
        global_C = torch.matmul(global_A, global_B)

    n_iterations = 10

    durations = []

    for i in range(n_iterations):
        dist.barrier();
        begin = time.time()

        if rank == 0:
            print(f"Calling MatMul...")

        # Perform distributed matrix multiplication
        dt_C = torch.matmul(dt_A, dt_B)

        # Ensure the computation is complete.
        # Check if we need to materialize the matrix,
        # and if so, what the new placement should be.
        requires_redistribute,new_placements = util.materialized_placements(dt_C.device_mesh, dt_C.placements)

        if requires_redistribute:
            dt_C = dt_C.redistribute(dt_C.device_mesh, new_placements)

        dist.barrier();
        end = time.time()

        duration = end - begin
        durations.append(duration)

        if VERIFY_RESULT:
            test_C = dt_C.full_tensor()
            # Verify results match within numerical tolerance
            rtol = 1e-5
            atol = 1e-8
            is_close = torch.allclose(test_C, global_C, rtol=rtol, atol=atol)

            if is_close:
                if rank == 0:
                    print(f"OK!")
            else:
                if rank == 0:
                    print(f"FAILED!")
                max_diff = torch.max(torch.abs(test_C - global_C))
                print(f"Process {rank}: Maximum absolute difference: {max_diff}")
                break
                sys.exit(1)

        if rank == 0:
            gflops = 1e-9 * ((2 * m * n * k) + (3 * m * n))
            gflops_s = gflops / duration

            print(f"Achieved {gflops_s} GFLOPs")

    if rank == 0:
        print(f"Output is {dt_C}")

    if rank == 0:
        gflops = 1e-9 * ((2 * m * n * k) + (3 * m * n))
        all_gflops = [gflops / duration for duration in durations]
        print(f"Max GFLOPs: {np.max(all_gflops)}")
        print(f"Median GFLOPs: {np.median(all_gflops)}")

    dist.barrier()

    # Clean up
    dist.destroy_process_group()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Distributed matrix multiplication benchmark')
    parser.add_argument('--m', type=int, default=8*1024, help='Number of rows in matrix A')
    parser.add_argument('--n', type=int, default=48*1024, help='Number of columns in matrix B')
    parser.add_argument('--k', type=int, default=12*1024, help='Number of columns in A / rows in B')
    parser.add_argument('--a-partition', type=str, choices=['row', 'column', 'block'], default='row',
                      help='Partitioning strategy for matrix A')
    parser.add_argument('--b-partition', type=str, choices=['row', 'column', 'block'], default='row',
                      help='Partitioning strategy for matrix B')
    parser.add_argument('--replication', type=int, default=1,
                      help='Replication factor for both matrices')
    args = parser.parse_args()

    # Set up environment variables if not already set
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "RANK" not in os.environ:
        os.environ["RANK"] = util.get_rank_from_env()
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = util.get_nprocs_from_env()

    run_matmul_benchmark(
        args.m, args.n, args.k,
        args.a_partition, args.b_partition,
        args.replication
    )

if __name__ == "__main__":
    main()

