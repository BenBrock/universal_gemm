import math
from typing import Tuple

import torch
import nvshmem.core as nvshmem
from cuda.core.experimental._memory import Buffer


class NvshmemHeap:
    """
    Simple bump allocator backed by a single NVSHMEM tensor.
    """

    def __init__(self, capacity_elements: int, dtype: torch.dtype) -> None:
        if capacity_elements <= 0:
            raise ValueError("capacity_elements must be positive")
        self._dtype = dtype
        self._capacity = capacity_elements
        self._buffer = nvshmem.tensor((capacity_elements,), dtype=dtype)
        self._base_ptr = int(self._buffer.data_ptr())
        self._element_size = torch.tensor([], dtype=dtype).element_size()
        self._heap_ptr = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def reset(self) -> None:
        self._heap_ptr = 0

    def alloc(self, shape: Tuple[int, int] | Tuple[int, ...]) -> torch.Tensor:
        numel = int(math.prod(shape))
        if numel == 0:
            return self._buffer.narrow(0, 0, 0).view(shape)
        end = self._heap_ptr + numel
        if end > self._capacity:
            # Wrap to the start of the heap (no deallocation tracking yet).
            self._heap_ptr = 0
            end = numel
        if end > self._capacity:
            raise RuntimeError(
                f"NvshmemHeap exhausted: requested={numel} elements, "
                f"capacity={self._capacity} elements"
            )
        view = self._buffer[self._heap_ptr:end].view(shape)
        byte_offset = self._heap_ptr * self._element_size
        byte_size = numel * self._element_size
        view._nvshmem_buf = Buffer.from_handle(self._base_ptr + byte_offset, byte_size)
        self._heap_ptr = end
        return view

    def close(self) -> None:
        nvshmem.free_tensor(self._buffer)
