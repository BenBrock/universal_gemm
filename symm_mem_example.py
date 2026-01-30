#!/usr/bin/env python3
# example.py
import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, distribute_tensor
import torch.distributed._symmetric_memory as symm_mem

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

    if not symm_mem.is_nvshmem_available():
        raise RuntimeError("NVSHMEM backend not available in this build/system")

    symm_mem.set_backend("NVSHMEM")

    n = 1024
    buf = symm_mem.empty(n, device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, group)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()