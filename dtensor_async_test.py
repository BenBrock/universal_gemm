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

    m = 32*1024
    n = 32*1024
    k = 32*1024

    '''
    NOTE: I *think* things should work for uneven tile sizes?
    assert m % world_size == 0, "m must be divisible by world_size for even row sharding"
    assert k % world_size == 0, "k must be divisible by world_size for even row sharding"
    '''

    a_p = benchmark.util.row_partitioning()

    global_a = torch.randn(m, k, dtype=torch.float32, device=device)*50 + 50

    local_a = torch.randn(10, 10, dtype=torch.float32, device=device)*50 + 50
    local_b = torch.randn(10, 10, dtype=torch.float32, device=device)*50 + 50
    local_c = torch.randn(10, 10, dtype=torch.float32, device=device)*50 + 50

    dt_a = distribute_tensor(global_a, *a_p)

    tile_shape = dt.tile_shape(dt_a)
    tile_numel = tile_shape[0] * tile_shape[1]
    tiles_per_round = dt.grid_shape(dt_a)[0] * dt.grid_shape(dt_a)[1]
    dt.init_scratch(tile_numel * tiles_per_round, dt_a.dtype)

    dist.barrier()

    for _ in range(10):
        torch.addmm(local_c, local_a, local_b, out=local_c)

    comm_issue = 0
    comm_sync = 0
    compute = 0

    if rank == 0:
        for i in range(dt.grid_shape(dt_a)[0]):
            for k in range(dt.grid_shape(dt_a)[1]):
                begin = time.time()
                a_f = dt.get_tile_async(dt_a, (i,k))
                end = time.time()
                comm_issue += end - begin

                begin = time.time()
                torch.addmm(local_c, local_a, local_b, out=local_c)
                torch.cuda.current_stream().synchronize()
                end = time.time()
                compute += end - begin

                begin = time.time()
                a_t = a_f.get()
                end = time.time()
                comm_sync += end - begin

    begin = time.time()
    torch.cuda.current_stream().synchronize()
    end = time.time()
    torch_synchronize = end - begin

    begin = time.time()
    dt._get_async_nvshmem_stream().sync()
    end = time.time()
    nvshmem_synchronize = end - begin


    if rank == 0:
        print(f'comm_issue: {comm_issue}')
        print(f'comm_sync: {comm_sync}')
        print(f'compute: {compute}')
        print(f'Torch synchronize: {torch_synchronize}')
        print(f'NVSHMEM synchronize: {nvshmem_synchronize}')

    nvshmem.barrier_all(stream=stream)
    dist.barrier()

    nvshmem.free_tensor(dt_a.nvshmem_base())
    dt.free_get_tile_scratch()
    nvshmem.finalize()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
