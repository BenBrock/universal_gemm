#!/usr/bin/env python3
# example.py
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

import benchmark.util
import nvshmem.core
from cuda.core.experimental import Device

def main():
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

    uid = nvshmem.core.get_unique_id(empty=(local_rank != 0))
    uid_bytes = uid._data.view(np.uint8).copy()
    uid_tensor = torch.from_numpy(uid_bytes).cuda()
    dist.broadcast(uid_tensor, src=0)
    dist.barrier()
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)
    nvshmem.core.init(
        device=dev,
        uid=uid,
        rank=local_rank,
        nranks=world_size,
        initializer_method="uid",
    )

    m = 64
    n = 64
    assert m % world_size == 0, "m must be divisible by world_size for even row sharding"
    local_shape = (m // world_size, n)

    a_p = benchmark.util.row_partitioning()
    device_mesh, placements = a_p

    # Allocate local shard via NVSHMEM so we can use nvshmem4py peer tensors.
    local_a = nvshmem.core.tensor(local_shape, dtype=torch.float32)
    local_a.fill_(0.0)
    dt_a = DTensor.from_local(local_a, device_mesh, placements, run_check=False)

    if rank == 0:
        lt = local_a
        print(lt)
        print(lt.device, lt.dtype, lt.shape, lt.storage().data_ptr())

        lt.fill_(42.0)
        torch.cuda.synchronize()
        nvshmem_stream = nvshmem.core.NvshmemStream(torch.cuda.current_stream())
        for peer in range(world_size):
            if peer == rank:
                continue
            peer_tensor = nvshmem.core.get_peer_tensor(lt, peer)
            # Explicit NVSHMEM put to write into the remote tensor.
            nvshmem.core.put(peer_tensor, lt, peer, stream=nvshmem_stream)
        nvshmem.core.quiet(stream=nvshmem_stream)
        torch.cuda.current_stream().synchronize()

        if hasattr(nvshmem.core, "barrier_all"):
            nvshmem.core.barrier_all(stream=nvshmem_stream)
        elif hasattr(nvshmem.core, "barrier"):
            nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=nvshmem_stream)

    dist.barrier()
    if rank != 0:
        lt = local_a
        print(f"Rank {rank} local tensor after put: {lt.flatten()[:4].tolist()}")

    dist.barrier()

    nvshmem.core.finalize()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
