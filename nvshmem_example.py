#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.distributed as dist
import nvshmem.core as nvshmem
from cuda.core.experimental import Device


def main() -> None:
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus
    torch.cuda.set_device(gpu_id)

    print(f"Rank {rank}/{world_size} running on GPU {gpu_id}/{num_gpus}")

    # NVSHMEM init using UID broadcast
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
        rank=rank,
        nranks=world_size,
        initializer_method="uid",
    )

    dist.barrier()

    size = 1024

    tensor = nvshmem.interop.torch.empty((size,), dtype=torch.float32)

    nvshmem.free_tensor(tensor)

    nvshmem.finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
