#!/usr/bin/env python3
import os
import time
import numpy as np
import nvshmem.core
from cuda.core.experimental import Device


def _env_int(name: str) -> int:
    val = os.environ.get(name)
    if val is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return int(val)


def _uid_path() -> str:
    return os.environ.get("NVSHMEM_UID_PATH", "/tmp/nvshmem_uid.bin")


def main() -> None:
    rank = _env_int("RANK")
    world_size = _env_int("WORLD_SIZE")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    dev = Device(local_rank)
    dev.set_current()

    uid_path = _uid_path()
    if rank == 0:
        uid = nvshmem.core.get_unique_id(empty=False)
        uid_bytes = uid._data.view(np.uint8).copy()
        with open(uid_path, "wb") as f:
            f.write(uid_bytes.tobytes())
    else:
        # Wait for UID file to appear
        deadline = time.time() + 30.0
        while not os.path.exists(uid_path):
            if time.time() > deadline:
                raise RuntimeError(f"Timed out waiting for UID file at {uid_path}")
            time.sleep(0.05)
        with open(uid_path, "rb") as f:
            uid_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        uid = nvshmem.core.get_unique_id(empty=True)
        uid._data[:] = uid_bytes.view(uid._data.dtype)

    nvshmem.core.init(
        device=dev,
        uid=uid,
        rank=local_rank,
        nranks=world_size,
        initializer_method="uid",
    )

    print(f"NVSHMEM initialized on rank {rank}/{world_size}")

    nvshmem.core.finalize()


if __name__ == "__main__":
    main()
