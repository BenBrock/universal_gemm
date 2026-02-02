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


    m = 64
    n = 64
    k = 64
    assert m % world_size == 0, "m must be divisible by world_size for even row sharding"
    assert k % world_size == 0, "k must be divisible by world_size for even row sharding"

    a_p = benchmark.util.row_partitioning()

    global_a = torch.randn(m, k, dtype=torch.float32)

    dt_a = distribute_tensor(global_a, *a_p)

    # Allocate local shard via NVSHMEM so we can use nvshmem4py peer tensors.
    local_a = nvshmem.tensor((m, k), dtype=torch.float32)

    local_a.fill_(42)

    nvshmem.free_tensor(dt_a.nvshmem_base())
    nvshmem.free_tensor(local_a);
    nvshmem.finalize()
    dist.destroy_process_group()
    return

    if rank == 0:
        lt = local_a
        print(lt)
        print(lt.device, lt.dtype, lt.shape, lt.storage().data_ptr())

        lt.fill_(42.0)
        torch.cuda.synchronize()
        nvshmem_stream = nvshmem.NvshmemStream(torch.cuda.current_stream())
        for peer in range(world_size):
            if peer == rank:
                continue
            peer_tensor = nvshmem.get_peer_tensor(lt, peer)
            # Explicit NVSHMEM put to write into the remote tensor.
            nvshmem.put(peer_tensor, lt, peer, stream=nvshmem_stream)
        nvshmem.quiet(stream=nvshmem_stream)
        torch.cuda.current_stream().synchronize()

        if hasattr(nvshmem, "barrier_all"):
            nvshmem.barrier_all(stream=nvshmem_stream)
        elif hasattr(nvshmem, "barrier"):
            nvshmem.barrier(nvshmem.Teams.TEAM_WORLD, stream=nvshmem_stream)

    dist.barrier()
    if rank != 0:
        lt = local_a
        print(f"Rank {rank} local tensor after put: {lt.flatten()[:4].tolist()}")

    dist.barrier()

    nvshmem.free_tensor(local_a);
    nvshmem.finalize()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
