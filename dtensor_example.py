#!/usr/bin/env python3
# example.py
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

import benchmark.util
import dtensor_mm_handler
import dtensor_utils
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

    m = 1024
    n = 1024
    k = 1024

    '''
    NOTE: I *think* things should work for uneven tile sizes?
    assert m % world_size == 0, "m must be divisible by world_size for even row sharding"
    assert k % world_size == 0, "k must be divisible by world_size for even row sharding"
    '''

    a_p = benchmark.util.row_partitioning()
    b_p = benchmark.util.row_partitioning()
    c_p = benchmark.util.row_partitioning()

    global_a = torch.randn(m, k, dtype=torch.float32)*50 + 50
    global_b = torch.randn(k, n, dtype=torch.float32)*50 + 50
    global_c = torch.zeros(m, n, dtype=torch.float32)*50 + 50

    dt_a = distribute_tensor(global_a, *a_p)
    dt_b = distribute_tensor(global_b, *b_p)
    dt_c = distribute_tensor(global_c, *c_p)

    dist.barrier()
    torch.matmul(dt_a, dt_b, out=dt_c)
    dist.barrier()

    torch.matmul(global_a, global_b, out=global_c)
    full_c = dt_c.full_tensor()
    if rank == 0:
        torch.testing.assert_close(
            full_c.cpu(),
            global_c.cpu(),
            rtol=1e-4,
            atol=1e-5,
        )

    nvshmem.free_tensor(dt_a.nvshmem_base())
    nvshmem.free_tensor(dt_b.nvshmem_base())
    nvshmem.free_tensor(dt_c.nvshmem_base())
    nvshmem.finalize()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
