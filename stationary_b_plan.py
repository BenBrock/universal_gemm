from stationary_c_plan import MultiplyOp

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from tile_bounds import Slice1D, Slice2D, overlapping_tiles, tile_bounds


def _format_slice_1d(s: Slice1D) -> str:
    return f"{s.start}:{s.stop}"


def _format_slice_2d(s: Slice2D) -> str:
    return f"[{_format_slice_1d(s.rows)}, {_format_slice_1d(s.cols)}]"


def format_stationary_b_op(op: MultiplyOp) -> str:
    return (
        f"C.tile{op.c_idx}{_format_slice_2d(op.c_slice)} += "
        f"A.tile{op.a_idx}{_format_slice_2d(op.a_slice)} * "
        f"B.tile{op.b_idx}{_format_slice_2d(op.b_slice)}"
    )


def format_stationary_b_plan(ops: list[MultiplyOp], rank: int | None = None) -> str:
    prefix = f"Stationary-B plan (rank={rank})" if rank is not None else "Stationary-B plan"
    lines = [f"{prefix}: {len(ops)} ops"]
    for idx, op in enumerate(ops):
        lines.append(f"  {idx:04d}: {format_stationary_b_op(op)}")
    return "\n".join(lines)


def _replica_m_range(
    *,
    m_extent: int,
    rep_factor: int,
    replica_idx: int,
) -> Slice1D:
    if m_extent < 0:
        raise ValueError(f"m_extent must be non-negative, got {m_extent}")
    if rep_factor <= 0:
        raise ValueError(f"rep_factor must be positive, got {rep_factor}")
    if replica_idx < 0 or replica_idx >= rep_factor:
        raise ValueError(
            f"replica_idx {replica_idx} out of range for rep_factor {rep_factor}"
        )

    chunk_size = (m_extent + rep_factor - 1) // rep_factor
    start = chunk_size * replica_idx
    stop = min(m_extent, chunk_size * (replica_idx + 1))
    return Slice1D(start, stop)


def build_stationary_b_ops(
    a: DTensor,
    b: DTensor,
    c: DTensor,
    *,
    local_only: bool = True,
) -> list[MultiplyOp]:
    """
    Build the Stationary-B local multiply-operation list.

    This mirrors drc_matrix's stationary-B construction:
      - iterate local B tiles
      - restrict M coverage by B replica index
      - find overlapping A and C tiles
      - emit clipped local tile slices for one multiply op
    """
    if not (a.ndim == 2 and b.ndim == 2 and c.ndim == 2):
        raise ValueError("build_stationary_b_ops expects rank-2 DTensors for a, b, and c")
    if a.shape[1] != b.shape[0] or c.shape[0] != a.shape[0] or c.shape[1] != b.shape[1]:
        raise ValueError(
            f"shape mismatch: a={tuple(a.shape)}, b={tuple(b.shape)}, c={tuple(c.shape)}"
        )
    import dtensor_utils as dt  # Lazy import to avoid circular import at module import time.

    rank = dist.get_rank() if dist.is_initialized() else 0
    b_grid_rows, b_grid_cols = dt.grid_shape(b)
    b_rep_factor = dt.replication_factor(b)
    b_replica = dt.my_replica(b)

    global_i_bounds = _replica_m_range(
        m_extent=int(c.shape[0]),
        rep_factor=b_rep_factor,
        replica_idx=b_replica,
    )
    if global_i_bounds.size == 0:
        return []

    ops: list[MultiplyOp] = []

    for k in range(b_grid_rows):
        for j in range(b_grid_cols):
            b_idx = (k, j)
            if local_only and dt.tile_rank(b, b_idx) != rank:
                continue

            b_bounds = tile_bounds(b, b_idx)
            a_region = Slice2D(global_i_bounds, b_bounds.rows)
            a_tiles = overlapping_tiles(a, a_region)

            for a_idx in a_tiles:
                a_bounds = dt.intersect_2d(tile_bounds(a, a_idx), a_region)
                c_region = Slice2D(a_bounds.rows, b_bounds.cols)
                c_tiles = overlapping_tiles(c, c_region)

                for c_idx in c_tiles:
                    c_bounds = tile_bounds(c, c_idx)

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


def execute_stationary_b_ops(
    a: DTensor,
    b: DTensor,
    c: DTensor,
    ops: list[MultiplyOp],
    *,
    release_remote_tiles: bool = True,
) -> None:
    """
    Execute a previously generated Stationary-B op list.

    For each op:
      - fetch A tile (possibly remote)
      - use local B tile (stationary)
      - compute local partial product
      - one-sided accumulate into C tile slice (local or remote)
    """
    import dtensor_utils as dt  # Lazy import to avoid circular import at module import time.

    rank = dist.get_rank() if dist.is_initialized() else 0

    for op in ops:
        _validate_op_shapes(op)

        owner = dt.tile_rank(b, op.b_idx)
        if owner != rank:
            raise RuntimeError(
                f"Stationary-B execution expected local b_idx={op.b_idx}, "
                f"owner={owner}, rank={rank}"
            )

        a_tile = dt.get_tile(a, op.a_idx)
        b_tile = dt.tile(b, op.b_idx)

        a_view = a_tile[*op.a_slice.as_slices()]
        b_view = b_tile[*op.b_slice.as_slices()]
        prod = torch.mm(a_view, b_view)

        dt.accumulate_tile(
            c,
            op.c_idx,
            prod,
            slice_=op.c_slice,
        ).wait()

        # get_tile uses pooled scratch for remote tiles.
        # Synchronize before releasing/reusing scratch.
        if c.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

        if release_remote_tiles:
            dt.release_tile(a_tile)

    if c.device.type == "cuda":
        torch.cuda.current_stream().synchronize()


def execute_stationary_b(
    a: DTensor,
    b: DTensor,
    c: DTensor,
    *,
    local_only: bool = True,
    release_remote_tiles: bool = True,
) -> list[MultiplyOp]:
    """
    Build and execute a Stationary-B plan.
    Returns the generated op list for debugging/printing.
    """
    import dtensor_utils as dt  # Lazy import to avoid circular import at module import time.

    c_rep_factor = dt.replication_factor(c)
    if c_rep_factor > 1 and dt.my_replica(c) != 0:
        # We reduce replica-local C chunks into origin replica 0.
        # Clear non-origin local C each run to avoid stale carry-over.
        c.to_local().zero_()

    ops = build_stationary_b_ops(a, b, c, local_only=local_only)
    execute_stationary_b_ops(
        a,
        b,
        c,
        ops,
        release_remote_tiles=release_remote_tiles,
    )

    if c_rep_factor > 1:
        # Ensure all ranks finish local compute before one-sided reduction.
        if dist.is_initialized():
            dist.barrier()
        dt.reduce_replicas(c)
        # Wait for all one-sided updates into origin replica to complete.
        if dist.is_initialized():
            dist.barrier()
    return ops
