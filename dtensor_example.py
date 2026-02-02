#!/usr/bin/env python3
# example.py
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor

import benchmark.util
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

    m = 64
    n = 64
    k = 64
    assert m % world_size == 0, "m must be divisible by world_size for even row sharding"
    assert k % world_size == 0, "k must be divisible by world_size for even row sharding"

    a_p = benchmark.util.row_partitioning()

    global_a = torch.randn(m, k, dtype=torch.float32)

    dt_a = distribute_tensor(global_a, *a_p)

    # Allocate a symmetric buffer matching the local shard size.
    local_shape = dt_a.to_local().shape
    local_a = nvshmem.tensor(local_shape, dtype=torch.float32)

    local_a.fill_(42)

    remote_base = dt_a.nvshmem_base()
    nvshmem.barrier_all(stream=stream)
    torch.cuda.synchronize()
    if rank == 1:
        nvshmem.put(remote_base, local_a, 0, stream=stream)
        nvshmem.quiet(stream=stream)
        torch.cuda.synchronize()

    nvshmem.barrier_all(stream=stream)
    torch.cuda.synchronize()

    if rank == 0:
        print(dt_a.to_local())

    nvshmem.barrier_all(stream=stream)
    torch.cuda.synchronize()

    nvshmem.free_tensor(dt_a.nvshmem_base())
    nvshmem.free_tensor(local_a);
    nvshmem.finalize()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
