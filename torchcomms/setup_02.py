#!/usr/bin/env python3
# example.py
import torch
# from torchcomms import new_comm, ReduceOp
import torchcomms

def main():
    # Initialize TorchComm with NCCLX backend
    device = torch.device("cuda")
    comm = torchcomms.new_comm("ncclx", device, name="main_comm")

    # Get rank and world size
    rank = comm.get_rank()
    world_size = comm.get_size()

    # Calculate device ID
    num_devices = torch.cuda.device_count()
    device_id = rank % num_devices
    target_device = torch.device(f"cuda:{device_id}")

    print(f"Rank {rank}/{world_size}: Running on device {device_id}")

    size = 1024
    dtype = torch.float32

    allocator = torchcomms.get_mem_allocator(comm.get_backend())

    pool = torch.cuda.MemPool(allocator)

    with torch.cuda.use_mem_pool(pool):
        win_buf = torch.ones(
            [size], dtype=dtype, device=device
        )

    print(comm.get_backend())
    print(allocator)

    print(win_buf)

    comm.barrier(False)

    win = comm.new_window()
    win.tensor_register(win_buf)
    comm.barrier(False)

    win.tensor_deregister()

    # Cleanup
    comm.finalize()

if __name__ == "__main__":
    main()
