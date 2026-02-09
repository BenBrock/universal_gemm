from dataclasses import dataclass

from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard


@dataclass(frozen=True)
class Slice1D:
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < 0:
            raise ValueError(f"Slice1D expects non-negative bounds, got [{self.start}, {self.stop})")
        if self.stop < self.start:
            raise ValueError(f"Slice1D expects stop >= start, got [{self.start}, {self.stop})")

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class Slice2D:
    rows: Slice1D
    cols: Slice1D

    @property
    def row_slice(self) -> Slice1D:
        return self.rows

    @property
    def col_slice(self) -> Slice1D:
        return self.cols

    def as_slices(self) -> tuple[slice, slice]:
        return (self.rows.as_slice(), self.cols.as_slice())

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows.size, self.cols.size)


def _grid_shape(dt: DTensor) -> tuple[int, int]:
    if dt.ndim != 2:
        raise ValueError(f"grid_shape expects a 2D DTensor, got ndim={dt.ndim}")

    grid_rows = 1
    grid_cols = 1
    mesh = dt.device_mesh
    for mesh_dim, placement in enumerate(dt.placements):
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


def tile_bounds(dt: DTensor, coord: tuple[int, int]) -> Slice2D:
    """
    Return global 2D half-open bounds for a tile coordinate.

    The returned bounds are suitable for indexing as:
        dt.full_tensor()[*tile_bounds(dt, coord).as_slices()]
    """
    if dt.ndim != 2:
        raise ValueError(f"tile_bounds expects a 2D DTensor, got ndim={dt.ndim}")

    grid_rows, grid_cols = _grid_shape(dt)
    row_idx, col_idx = coord
    if not (0 <= row_idx < grid_rows and 0 <= col_idx < grid_cols):
        raise ValueError(
            f"tile coord {coord} out of range for grid {(grid_rows, grid_cols)}"
        )

    m, n = dt.shape
    max_rows = (m + grid_rows - 1) // grid_rows
    max_cols = (n + grid_cols - 1) // grid_cols

    row_start = row_idx * max_rows
    col_start = col_idx * max_cols
    row_stop = min(row_start + max_rows, m)
    col_stop = min(col_start + max_cols, n)

    return Slice2D(
        rows=Slice1D(row_start, row_stop),
        cols=Slice1D(col_start, col_stop),
    )


def overlapping_tiles(dt: DTensor, region: Slice2D) -> list[tuple[int, int]]:
    """
    Return tile coordinates that overlap with a global 2D region.

    Overlap is computed on half-open intervals; a non-empty intersection on both
    axes is required for a tile to be included.
    """
    if dt.ndim != 2:
        raise ValueError(f"overlapping_tiles expects a 2D DTensor, got ndim={dt.ndim}")

    m, n = dt.shape
    # Clamp region to the valid global extent.
    row_lo = max(0, region.rows.start)
    row_hi = min(m, region.rows.stop)
    col_lo = max(0, region.cols.start)
    col_hi = min(n, region.cols.stop)

    if row_lo >= row_hi or col_lo >= col_hi:
        return []

    grid_rows, grid_cols = _grid_shape(dt)
    max_rows = (m + grid_rows - 1) // grid_rows
    max_cols = (n + grid_cols - 1) // grid_cols

    row_start_idx = max(0, min(grid_rows - 1, row_lo // max_rows))
    row_end_idx = max(0, min(grid_rows - 1, (row_hi - 1) // max_rows))
    col_start_idx = max(0, min(grid_cols - 1, col_lo // max_cols))
    col_end_idx = max(0, min(grid_cols - 1, (col_hi - 1) // max_cols))

    coords: list[tuple[int, int]] = []
    for i in range(row_start_idx, row_end_idx + 1):
        for j in range(col_start_idx, col_end_idx + 1):
            coords.append((i, j))
    return coords
