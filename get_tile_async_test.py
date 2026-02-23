#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.distributed as dist

import benchmark.util
import dtensor_utils as dt
import nvshmem.core as nvshmem
from cuda.core.experimental import Device
from torch.distributed.tensor import distribute_tensor


def main() -> None:
    dist.init_process_group(backend="nccl")

    assert os.environ.get("TORCH_DTENSOR_USE_NVSHMEM", "0") == "1", (
        "Expected TORCH_DTENSOR_USE_NVSHMEM=1 for this example."
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus
    torch.cuda.set_device(gpu_id)

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
    a_p = benchmark.util.row_partitioning()

    global_a = torch.randn(m, n, dtype=torch.float32)
    dt_a = distribute_tensor(global_a, *a_p)

    tile_shape = dt.tile_shape(dt_a)
    tile_numel = tile_shape[0] * tile_shape[1]
    tiles_per_round = dt.grid_shape(dt_a)[0] * dt.grid_shape(dt_a)[1]
    dt.init_scratch(tile_numel * tiles_per_round, dt_a.dtype)

    if rank == 0:
        grid = dt.grid_shape(dt_a)
        futures = []
        for i in range(grid[0]):
            for j in range(grid[1]):
                futures.append((i, j, dt.get_tile_async(dt_a, (i, j))))
        for i, j, fut in futures:
            tile = fut.get()
            expected_shape = dt.tile_shape(dt_a, (i, j))
            print(tile)
            assert tuple(tile.shape) == expected_shape, (
                f"tile {i, j} shape {tuple(tile.shape)} != {expected_shape}"
            )
            print(f"tile ({i}, {j}) ok: {tuple(tile.shape)}")

    dist.barrier()
    nvshmem.free_tensor(dt_a.nvshmem_base())
    dt.free_get_tile_scratch()
    nvshmem.finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
