import torch
import torch.distributed as dist
import nvshmem.core as nvshmem

from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard


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


def _tile_owner_rank(dt: DTensor, coord: tuple[int, int]) -> int:
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
    owner_rank = _tile_owner_rank(dt, coord)

    local_buf = torch.empty(
        max_shape, dtype=dt.dtype, device=dt.device  # type: ignore[arg-type]
    )
    if dist.is_initialized() and owner_rank == dist.get_rank():
        local_buf.copy_(dt.nvshmem_base())
        torch.cuda.synchronize()
    else:
        remote_buf = nvshmem.get_peer_tensor(dt.nvshmem_base(), owner_rank)
        local_buf.copy_(remote_buf)
        torch.cuda.synchronize()

    rows, cols = actual_shape
    return local_buf[:rows, :cols]
