#!/usr/bin/env python3
# example.py
import os
import torch
import torchcomms

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = local_rank
    torch.cuda.set_device(device_id)

    device = torch.device(f"cuda:{device_id}")
    comm = torchcomms.new_comm("ncclx", device, name="main_comm")

    # Get rank and world size
    rank = comm.get_rank()
    world_size = comm.get_size()

    # Calculate device ID
    num_devices = torch.cuda.device_count()

    print(f"Rank {rank}/{world_size}: Running on device {device_id}/{num_devices}")

    size = 1024
    dtype = torch.float32

    buffer = comm.mem_allocator.allocate(size, dtype, device)

    # Cleanup
    comm.finalize()

if __name__ == "__main__":
    main()
