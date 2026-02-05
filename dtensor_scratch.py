import threading
import weakref
from typing import Dict, List, Tuple

import torch
import nvshmem.core as nvshmem


class NvshmemTensorPool:
    """
    Pool of NVSHMEM-backed tensors allocated collectively once.

    Tensors are returned to the free list via weakref.finalize when GC'd,
    or explicitly via free().
    """

    def __init__(self, shape: Tuple[int, int], dtype: torch.dtype, slots: int) -> None:
        self._lock = threading.Lock()
        self._shape = shape
        self._dtype = dtype
        self._tensors: List[torch.Tensor] = [nvshmem.tensor(shape, dtype=dtype) for _ in range(slots)]
        self._free: List[int] = list(range(slots))
        self._in_use: Dict[int, int] = {}

    def alloc(self) -> torch.Tensor:
        with self._lock:
            if not self._free:
                raise RuntimeError("NvshmemTensorPool exhausted")
            idx = self._free.pop()
            t = self._tensors[idx]
            self._in_use[id(t)] = idx
            # Ensure finalizer stays alive by attaching to the tensor object.
            t._pool_finalizer = weakref.finalize(t, self._return_to_pool, idx)
            return t

    def free(self, tensor: torch.Tensor) -> None:
        idx = self._in_use.pop(id(tensor), None)
        if idx is None:
            return
        with self._lock:
            self._free.append(idx)

    def _return_to_pool(self, idx: int) -> None:
        with self._lock:
            self._free.append(idx)

    def close(self) -> None:
        for t in self._tensors:
            nvshmem.free_tensor(t)
        self._tensors.clear()
        self._free.clear()
        self._in_use.clear()
