from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from tile_bounds import Slice1D, Slice2D, overlapping_tiles, tile_bounds


@dataclass(frozen=True)
class MultiplyOp:
    a_idx: tuple[int, int]
    b_idx: tuple[int, int]
    c_idx: tuple[int, int]
    a_slice: Slice2D
    b_slice: Slice2D
    c_slice: Slice2D


def _format_slice_1d(s: Slice1D) -> str:
    return f"{s.start}:{s.stop}"


def _format_slice_2d(s: Slice2D) -> str:
    return f"[{_format_slice_1d(s.rows)}, {_format_slice_1d(s.cols)}]"


def format_op(op: MultiplyOp) -> str:
    return (
        f"C.tile{op.c_idx}{_format_slice_2d(op.c_slice)} += "
        f"A.tile{op.a_idx}{_format_slice_2d(op.a_slice)} * "
        f"B.tile{op.b_idx}{_format_slice_2d(op.b_slice)}"
    )


def format_plan(ops: list[MultiplyOp], rank: int | None = None) -> str:
    prefix = f"Stationary-C plan (rank={rank})" if rank is not None else "Stationary-C plan"
    lines = [f"{prefix}: {len(ops)} ops"]
    for idx, op in enumerate(ops):
        lines.append(f"  {idx:04d}: {format_op(op)}")
    return "\n".join(lines)


def build_stationary_c_ops(
    a: DTensor,
    b: DTensor,
    c: DTensor,
    *,
    local_only: bool = True,
) -> list[MultiplyOp]:
    """
    Build the Stationary-C local multiply-operation list (Algorithm 1).
    """
    if not (a.ndim == 2 and b.ndim == 2 and c.ndim == 2):
        raise ValueError("build_stationary_c_ops expects rank-2 DTensors for a, b, and c")
    if a.shape[1] != b.shape[0] or c.shape[0] != a.shape[0] or c.shape[1] != b.shape[1]:
        raise ValueError(
            f"shape mismatch: a={tuple(a.shape)}, b={tuple(b.shape)}, c={tuple(c.shape)}"
        )
    import dtensor_utils as dt  # Lazy import to avoid circular import at module import time.

    rank = dist.get_rank() if dist.is_initialized() else 0
    c_grid_rows, c_grid_cols = dt.grid_shape(c)

    ops: list[MultiplyOp] = []

    for i in range(c_grid_rows):
        for j in range(c_grid_cols):
            c_idx = (i, j)
            if local_only and dt.tile_rank(c, c_idx) != rank:
                continue

            c_bounds = tile_bounds(c, c_idx)
            a_region = Slice2D(c_bounds.rows, Slice1D(0, int(a.shape[1])))
            a_tiles = overlapping_tiles(a, a_region)

            for a_idx in a_tiles:
                a_bounds = tile_bounds(a, a_idx)
                b_region = Slice2D(a_bounds.cols, c_bounds.cols)
                b_tiles = overlapping_tiles(b, b_region)

                for b_idx in b_tiles:
                    b_bounds = tile_bounds(b, b_idx)

                    m_bounds = dt.intersect_1d(a_bounds.rows, c_bounds.rows)
                    k_bounds = dt.intersect_1d(a_bounds.cols, b_bounds.rows)
                    n_bounds = dt.intersect_1d(b_bounds.cols, c_bounds.cols)

                    global_a = Slice2D(m_bounds, k_bounds)
                    global_b = Slice2D(k_bounds, n_bounds)
                    global_c = Slice2D(m_bounds, n_bounds)

                    if global_a.shape[0] == 0 or global_a.shape[1] == 0:
                        continue
                    if global_b.shape[0] == 0 or global_b.shape[1] == 0:
                        continue
                    if global_c.shape[0] == 0 or global_c.shape[1] == 0:
                        continue

                    ops.append(
                        MultiplyOp(
                            a_idx=a_idx,
                            b_idx=b_idx,
                            c_idx=c_idx,
                            a_slice=dt.subtract_offset(global_a, dt.tile_offset(a, a_idx)),
                            b_slice=dt.subtract_offset(global_b, dt.tile_offset(b, b_idx)),
                            c_slice=dt.subtract_offset(global_c, dt.tile_offset(c, c_idx)),
                        )
                    )
    return ops


def _validate_op_shapes(op: MultiplyOp) -> None:
    if op.a_slice.shape[0] != op.c_slice.shape[0]:
        raise RuntimeError(f"Bad op (M mismatch): {op}")
    if op.a_slice.shape[1] != op.b_slice.shape[0]:
        raise RuntimeError(f"Bad op (K mismatch): {op}")
    if op.b_slice.shape[1] != op.c_slice.shape[1]:
        raise RuntimeError(f"Bad op (N mismatch): {op}")


def execute_stationary_c_ops(
    a: DTensor,
    b: DTensor,
    c: DTensor,
    ops: list[MultiplyOp],
    *,
    release_remote_tiles: bool = True,
) -> None:
    """
    Execute a previously generated Stationary-C op list.

    This implementation is intentionally simple:
      - fetch A/B tiles via get_tile
      - slice views according to op slices
      - accumulate with addmm into local C tile
    """
    import dtensor_utils as dt  # Lazy import to avoid circular import at module import time.

    rank = dist.get_rank() if dist.is_initialized() else 0

    for op in ops:
        _validate_op_shapes(op)

        owner = dt.tile_rank(c, op.c_idx)
        if owner != rank:
            raise RuntimeError(
                f"Stationary-C execution expected local c_idx={op.c_idx}, "
                f"owner={owner}, rank={rank}"
            )

        a_tile = dt.get_tile(a, op.a_idx)
        b_tile = dt.get_tile(b, op.b_idx)
        c_tile = dt.tile(c, op.c_idx)

        a_view = a_tile[*op.a_slice.as_slices()]
        b_view = b_tile[*op.b_slice.as_slices()]
        c_view = c_tile[*op.c_slice.as_slices()]

        torch.addmm(c_view, a_view, b_view, out=c_view)
        # This simple executor reuses pooled scratch tiles from get_tile.
        # Synchronize before the next op so scratch buffers are not
        # overwritten while addmm is still in-flight on the stream.
        if c.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

        if release_remote_tiles:
            dt.release_tile(a_tile)
            dt.release_tile(b_tile)

    if c.device.type == "cuda":
        torch.cuda.current_stream().synchronize()


def execute_stationary_c(
    a: DTensor,
    b: DTensor,
    c: DTensor,
    *,
    local_only: bool = True,
    release_remote_tiles: bool = True,
) -> list[MultiplyOp]:
    """
    Build and execute a Stationary-C plan.
    Returns the generated op list for debugging/printing.
    """
    ops = build_stationary_c_ops(a, b, c, local_only=local_only)
    execute_stationary_c_ops(
        a,
        b,
        c,
        ops,
        release_remote_tiles=release_remote_tiles,
    )
    return ops
