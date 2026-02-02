#!/usr/bin/env python3
# example.py
import torch
# from torchcomms import new_comm, ReduceOp
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

    # Program

    if rank == 1:
        send_data = torch.zeros((10,), dtype=dtype, device=device)
        win.put(send_data, dst_rank=0, target_offset_nelems=0, async_op=False)
        win.signal(peer_rank=0, async_op=False)

    if rank == 0:
        win.wait_signal(peer_rank=1, async_op=False)
        print(win_buf)

    comm.barrier(False)
    win.tensor_deregister()

    # Cleanup
    comm.finalize()

if __name__ == "__main__":
    main()
