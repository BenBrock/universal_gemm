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
