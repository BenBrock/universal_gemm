import torch
import torch.distributed as dist
import nvshmem.core as nvshmem
from cuda.core.experimental import Device
from cuda.core.experimental._stream import Stream

from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from dtensor_scratch import NvshmemTensorPool

_tile_pool: NvshmemTensorPool | None = None
_get_tile_async_raw_nvshmem_stream: Stream | None = None


def _get_async_nvshmem_stream() -> Stream:
    global _get_tile_async_raw_nvshmem_stream
    if _get_tile_async_raw_nvshmem_stream is None:
        device_id = torch.cuda.current_device()
        _get_tile_async_raw_nvshmem_stream = Device(device_id).create_stream()
    return _get_tile_async_raw_nvshmem_stream


def init_get_tile_scratch(
    dt: DTensor,
    slots: int = 4,
) -> None:
    """
    Allocate a pool of NVSHMEM tensors for temporary tiles (collective).
    """
    global _tile_pool
    max_shape = tile_shape(dt)
    if _tile_pool is None:
        _tile_pool = NvshmemTensorPool(max_shape, dt.dtype, slots)


def free_get_tile_scratch() -> None:
    """
    Free the NVSHMEM pool (collective).
    """
    global _tile_pool
    if _tile_pool is not None:
        _tile_pool.close()
        _tile_pool = None


def grid_shape(dt: DTensor) -> tuple[int, int]:
    """
    Return the 2D tile grid shape implied by DTensor sharding.

    Replication and Partial placements do not affect the grid shape.
    Only Shard placements on dim 0/1 are considered.
    """
    if dt.ndim != 2:
        raise ValueError(f"grid_shape expects a 2D DTensor, got ndim={dt.ndim}")

    placements = dt.placements
    mesh = dt.device_mesh

    grid_rows = 1
    grid_cols = 1

    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            if placement.dim == 0:
                grid_rows *= mesh.size(mesh_dim)
            elif placement.dim == 1:
                grid_cols *= mesh.size(mesh_dim)
            else:
                raise ValueError(
                    f"grid_shape only supports Shard on dim 0/1, got dim={placement.dim}"
                )
        elif isinstance(placement, (Replicate, Partial)):
            continue
        else:
            raise ValueError(f"grid_shape does not support placement {placement}")

    return (grid_rows, grid_cols)


def tile_shape(dt: DTensor, coord: tuple[int, int] | None = None) -> tuple[int, int]:
    """
    Return the tile shape for a 2D DTensor.

    If coord is None, returns the maximal (non-edge-clipped) tile shape.
    If coord is provided, returns the exact shape for that tile.
    """
    if dt.ndim != 2:
        raise ValueError(f"tile_shape expects a 2D DTensor, got ndim={dt.ndim}")

    grid_rows, grid_cols = grid_shape(dt)
    m, n = dt.shape

    max_rows = (m + grid_rows - 1) // grid_rows
    max_cols = (n + grid_cols - 1) // grid_cols

    if coord is None:
        return (max_rows, max_cols)

    row_idx, col_idx = coord
    if not (0 <= row_idx < grid_rows and 0 <= col_idx < grid_cols):
        raise ValueError(
            f"tile coord {coord} out of range for grid {(grid_rows, grid_cols)}"
        )

    row_start = row_idx * max_rows
    col_start = col_idx * max_cols
    rows = max(min(max_rows, m - row_start), 0)
    cols = max(min(max_cols, n - col_start), 0)

    return (rows, cols)


def _unflatten_index(index: int, sizes: list[int]) -> list[int]:
    coords: list[int] = []
    for size in reversed(sizes):
        coords.append(index % size)
        index //= size
    coords.reverse()
    return coords


def tile_rank(dt: DTensor, coord: tuple[int, int]) -> int:
    placements = dt.placements
    mesh = dt.device_mesh

    row_dims = [
        mesh_dim
        for mesh_dim, placement in enumerate(placements)
        if isinstance(placement, Shard) and placement.dim == 0
    ]
    col_dims = [
        mesh_dim
        for mesh_dim, placement in enumerate(placements)
        if isinstance(placement, Shard) and placement.dim == 1
    ]

    if mesh.ndim == 1:
        shard_dim = None
        if row_dims:
            shard_dim = 0
            tile_index = coord[0]
            shard_sizes = [mesh.size(d) for d in row_dims]
        elif col_dims:
            shard_dim = 1
            tile_index = coord[1]
            shard_sizes = [mesh.size(d) for d in col_dims]
        else:
            shard_dim = None
            tile_index = 0
            shard_sizes = [mesh.size(0)]

        if shard_dim is None:
            owner_index = 0
        else:
            owner_index = _unflatten_index(tile_index, shard_sizes)[0]

        mesh_tensor = mesh.mesh.flatten()
        return int(mesh_tensor[owner_index].item())

    full_coord = [0] * mesh.ndim
    if row_dims:
        row_sizes = [mesh.size(d) for d in row_dims]
        row_coords = _unflatten_index(coord[0], row_sizes)
        for d, c in zip(row_dims, row_coords):
            full_coord[d] = c
    if col_dims:
        col_sizes = [mesh.size(d) for d in col_dims]
        col_coords = _unflatten_index(coord[1], col_sizes)
        for d, c in zip(col_dims, col_coords):
            full_coord[d] = c

    rank_map = mesh._layout.remap_to_tensor(mesh._rank_map)
    return int(rank_map[tuple(full_coord)].item())


def get_tile(dt: DTensor, coord: tuple[int, int]) -> torch.Tensor:
    """
    Copy a (possibly remote) DTensor tile into a local NVSHMEM tensor.
    Returns a view clipped to the precise tile shape for coord.
    """
    if dt.ndim != 2:
        raise ValueError(f"get_tile expects a 2D DTensor, got ndim={dt.ndim}")

    grid_rows, grid_cols = grid_shape(dt)
    row_idx, col_idx = coord
    if not (0 <= row_idx < grid_rows and 0 <= col_idx < grid_cols):
        raise ValueError(
            f"tile coord {coord} out of range for grid {(grid_rows, grid_cols)}"
        )

    max_shape = tile_shape(dt)
    actual_shape = tile_shape(dt, coord)
    owner_rank = tile_rank(dt, coord)

    local_buf = torch.empty(
        max_shape, dtype=dt.dtype, device=dt.device  # type: ignore[arg-type]
    )
    if _tile_pool is None:
        raise RuntimeError(
            "get_tile requires init_get_tile_scratch(dt) to be called collectively before use."
        )
    scratch = _tile_pool.alloc()
    torch_stream = torch.cuda.Stream()
    src = dt.nvshmem_base().view(max_shape)
    nvshmem.get(scratch, src, remote_pe=owner_rank, stream=torch_stream)
    nvshmem.quiet(torch_stream)
    torch_stream.synchronize()
    local_buf.copy_(scratch)
    _tile_pool.free(scratch)

    rows, cols = actual_shape
    return local_buf[:rows, :cols]


def tile(dt: DTensor, coord: tuple[int, int]) -> torch.Tensor:
    """
    Return a local view of the tile if owned by the calling rank.
    Raises if the tile is remote.
    """
    if dt.ndim != 2:
        raise ValueError(f"tile expects a 2D DTensor, got ndim={dt.ndim}")

    owner_rank = tile_rank(dt, coord)
    rank = dist.get_rank() if dist.is_initialized() else 0
    if owner_rank != rank:
        raise RuntimeError(
            f"tile {coord} is owned by rank {owner_rank}, current rank is {rank}"
        )

    rows, cols = tile_shape(dt, coord)
    local = dt.to_local()
    return local[:rows, :cols]


def release_tile(tile: torch.Tensor) -> None:
    """
    Return a pooled scratch tile to the free list (no-op if not pooled).
    """
    owner = getattr(tile, "_scratch_owner", None)
    _tile_pool.free(owner)


class _TileFuture:
    def __init__(
        self,
        scratch: torch.Tensor,
        nvshmem_stream: nvshmem.nvshmem_types.NvshmemStream,
        rows: int,
        cols: int,
    ) -> None:
        self._scratch = scratch
        self._nvshmem_stream = nvshmem_stream
        self._rows = rows
        self._cols = cols

    def get(self) -> torch.Tensor:
        nvshmem.quiet(self._nvshmem_stream)
        self._nvshmem_stream.sync()
        view = self._scratch[: self._rows, : self._cols]
        view._scratch_owner = self._scratch
        return view


def get_tile_async(dt: DTensor, coord: tuple[int, int]) -> _TileFuture:
    """
    Asynchronously fetch a tile. Returns a future with a .get() method.
    """
    if dt.ndim != 2:
        raise ValueError(f"get_tile_async expects a 2D DTensor, got ndim={dt.ndim}")

    grid_rows, grid_cols = grid_shape(dt)
    row_idx, col_idx = coord
    if not (0 <= row_idx < grid_rows and 0 <= col_idx < grid_cols):
        raise ValueError(
            f"tile coord {coord} out of range for grid {(grid_rows, grid_cols)}"
        )

    max_shape = tile_shape(dt)
    rows, cols = tile_shape(dt, coord)
    owner_rank = tile_rank(dt, coord)

    if _tile_pool is None:
        raise RuntimeError(
            "get_tile_async requires init_get_tile_scratch(dt) to be called collectively before use."
        )
    scratch = _tile_pool.alloc()
    nvshmem_stream = _get_async_nvshmem_stream()
    src = dt.nvshmem_base().view(max_shape)
    nvshmem.get(scratch, src, remote_pe=owner_rank, stream=nvshmem_stream)
    return _TileFuture(scratch, nvshmem_stream, rows, cols)
