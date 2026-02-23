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


def _rotated_replica_k_ranges(
    *,
    k_extent: int,
    rep_factor: int,
    replica_idx: int,
    tile_sum_idx: int,
    k_tile_width: int,
) -> list[Slice1D]:
    """
    Return this replica's K ranges for a stationary tile after applying
    tile-dependent rotation.

    The unrotated base chunk is:
      [chunk_size * replica_idx, min(k_extent, chunk_size * (replica_idx + 1)))
    where chunk_size = ceil(k_extent / rep_factor).

    We then rotate this interval by:
      offset = (tile_sum_idx * k_tile_width) % k_extent

    Rotation is done on a ring [0, k_extent), so the result may wrap and split
    into two ranges.
    """
    if k_extent <= 0:
        return []
    if rep_factor <= 0:
        raise ValueError(f"rep_factor must be positive, got {rep_factor}")
    if replica_idx < 0 or replica_idx >= rep_factor:
        raise ValueError(
            f"replica_idx {replica_idx} out of range for rep_factor {rep_factor}"
        )
    if k_tile_width <= 0:
        raise ValueError(f"k_tile_width must be positive, got {k_tile_width}")

    chunk_size = (k_extent + rep_factor - 1) // rep_factor
    base_start = chunk_size * replica_idx
    base_stop = min(k_extent, chunk_size * (replica_idx + 1))
    base_len = max(base_stop - base_start, 0)
    if base_len == 0:
        return []

    offset = (tile_sum_idx * k_tile_width) % k_extent
    start = (base_start + offset) % k_extent
    stop = start + base_len
    if stop <= k_extent:
        return [Slice1D(start, stop)]
    return [Slice1D(start, k_extent), Slice1D(0, stop - k_extent)]


def build_stationary_c_ops(
    a: DTensor,
    b: DTensor,
    c: DTensor,
    *,
    local_only: bool = True,
) -> list[MultiplyOp]:
    """
    Build the Stationary-C local multiply-operation list (Algorithm 1).

    If C is replicated with factor r>1, each replica computes only a contiguous
    1/r chunk of the global K dimension, keyed by my_replica(c). This mirrors
    replicated stationary-C chunking in drc_matrix.
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
    c_rep_factor = dt.replication_factor(c)
    c_replica = dt.my_replica(c)

    k_extent = int(a.shape[1])
    k_tile_width = dt.tile_shape(a)[1]

    ops: list[MultiplyOp] = []

    for i in range(c_grid_rows):
        for j in range(c_grid_cols):
            c_idx = (i, j)
            if local_only and dt.tile_rank(c, c_idx) != rank:
                continue

            c_bounds = tile_bounds(c, c_idx)
            k_ranges = _rotated_replica_k_ranges(
                k_extent=k_extent,
                rep_factor=c_rep_factor,
                replica_idx=c_replica,
                tile_sum_idx=i + j,
                k_tile_width=k_tile_width,
            )

            for k_bounds_for_tile in k_ranges:
                a_region = Slice2D(c_bounds.rows, k_bounds_for_tile)
                a_tiles = overlapping_tiles(a, a_region)

                for a_idx in a_tiles:
                    # Clip the A tile bounds to this replica's rotated K chunk
                    # so straddling tiles do not generate out-of-chunk work.
                    a_bounds = dt.intersect_2d(tile_bounds(a, a_idx), a_region)
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


def execute_stationary_c_ops_async(
    a: DTensor,
    b: DTensor,
    c: DTensor,
    ops: list[MultiplyOp],
    *,
    release_remote_tiles: bool = True,
    max_outstanding_prefetch: int = 1,
) -> None:
    """
    Execute a Stationary-C op list with async A/B tile prefetch.

    Notes:
      - Ops are executed in-order (no iteration offset).
      - Prefetch is bounded by `max_outstanding_prefetch`.
      - Retrieved tiles are cached and evicted after their final use.
    """
    import dtensor_utils as dt  # Lazy import to avoid circular import at module import time.

    if max_outstanding_prefetch < 0:
        raise ValueError(
            f"max_outstanding_prefetch must be non-negative, got {max_outstanding_prefetch}"
        )

    rank = dist.get_rank() if dist.is_initialized() else 0
    if not ops:
        return

    a_tile_requests: dict[tuple[int, int], object] = {}
    b_tile_requests: dict[tuple[int, int], object] = {}
    a_tiles: dict[tuple[int, int], torch.Tensor] = {}
    b_tiles: dict[tuple[int, int], torch.Tensor] = {}

    # Track which requests count against the outstanding prefetch budget.
    prefetched_requests: set[tuple[str, tuple[int, int]]] = set()

    # Remaining uses for each fetched tile so we can evict exactly after
    # the final consuming op.
    a_remaining: dict[tuple[int, int], int] = {}
    b_remaining: dict[tuple[int, int], int] = {}
    for op in ops:
        a_remaining[op.a_idx] = a_remaining.get(op.a_idx, 0) + 1
        b_remaining[op.b_idx] = b_remaining.get(op.b_idx, 0) + 1

    def _select_maps(
        matrix_name: str,
    ) -> tuple[dict[tuple[int, int], object], dict[tuple[int, int], torch.Tensor]]:
        if matrix_name == "a":
            return a_tile_requests, a_tiles
        if matrix_name == "b":
            return b_tile_requests, b_tiles
        raise RuntimeError(f"Unknown matrix name: {matrix_name}")

    def _enqueue_tile(matrix_name: str, idx: tuple[int, int], *, prefetch: bool) -> bool:
        requests, tiles = _select_maps(matrix_name)
        if idx in tiles or idx in requests:
            return False
        matrix = a if matrix_name == "a" else b
        requests[idx] = dt.get_tile_async(matrix, idx)
        if prefetch:
            prefetched_requests.add((matrix_name, idx))
        return True

    def _get_tile(matrix_name: str, idx: tuple[int, int]) -> torch.Tensor:
        requests, tiles = _select_maps(matrix_name)
        future = requests.pop(idx, None)
        if future is not None:
            tile = future.get()
            tiles[idx] = tile
        else:
            tile = tiles.get(idx)
            if tile is None:
                # Required, non-prefetched path.
                matrix = a if matrix_name == "a" else b
                tile = dt.get_tile_async(matrix, idx).get()
                tiles[idx] = tile
        prefetched_requests.discard((matrix_name, idx))
        return tiles[idx]

    def _prefetch_for_next_op(next_op: MultiplyOp | None) -> None:
        if next_op is None or max_outstanding_prefetch == 0:
            return
        for matrix_name, idx in (("a", next_op.a_idx), ("b", next_op.b_idx)):
            if len(prefetched_requests) >= max_outstanding_prefetch:
                break
            _enqueue_tile(matrix_name, idx, prefetch=True)

    for op_idx, op in enumerate(ops):
        _validate_op_shapes(op)

        owner = dt.tile_rank(c, op.c_idx)
        if owner != rank:
            raise RuntimeError(
                f"Stationary-C async execution expected local c_idx={op.c_idx}, "
                f"owner={owner}, rank={rank}"
            )

        a_tile = _get_tile("a", op.a_idx)
        b_tile = _get_tile("b", op.b_idx)

        next_op = ops[op_idx + 1] if op_idx + 1 < len(ops) else None
        _prefetch_for_next_op(next_op)

        c_tile = dt.tile(c, op.c_idx)
        a_view = a_tile[*op.a_slice.as_slices()]
        b_view = b_tile[*op.b_slice.as_slices()]
        c_view = c_tile[*op.c_slice.as_slices()]
        torch.addmm(c_view, a_view, b_view, out=c_view)

        a_remaining[op.a_idx] -= 1
        if a_remaining[op.a_idx] == 0:
            tile = a_tiles.pop(op.a_idx, None)
            if release_remote_tiles and tile is not None:
                dt.release_tile(tile)

        b_remaining[op.b_idx] -= 1
        if b_remaining[op.b_idx] == 0:
            tile = b_tiles.pop(op.b_idx, None)
            if release_remote_tiles and tile is not None:
                dt.release_tile(tile)

    # Drain any leftover prefetched requests before returning.
    for idx, future in a_tile_requests.items():
        tile = future.get()
        if release_remote_tiles:
            dt.release_tile(tile)
    for idx, future in b_tile_requests.items():
        tile = future.get()
        if release_remote_tiles:
            dt.release_tile(tile)

    if c.device.type == "cuda":
        torch.cuda.current_stream().synchronize()


def execute_stationary_c(
    a: DTensor,
    b: DTensor,
    c: DTensor,
    *,
    local_only: bool = True,
    release_remote_tiles: bool = True,
    use_async: bool | None = None,
    max_outstanding_prefetch: int = 1,
) -> list[MultiplyOp]:
    """
    Build and execute a Stationary-C plan.
    Returns the generated op list for debugging/printing.
    """
    import dtensor_utils as dt  # Lazy import to avoid circular import at module import time.
    import os

    c_rep_factor = dt.replication_factor(c)
    if c_rep_factor > 1 and dt.my_replica(c) != 0:
        # We reduce partial C contributions into origin replica 0. Clear
        # non-origin local C before each call so repeated addmm(..., out=c)
        # does not re-add stale non-origin values across iterations.
        c.to_local().zero_()

    ops = build_stationary_c_ops(a, b, c, local_only=local_only)
    if use_async is None:
        use_async = True
    if use_async:
        execute_stationary_c_ops_async(
            a,
            b,
            c,
            ops,
            release_remote_tiles=release_remote_tiles,
            max_outstanding_prefetch=max_outstanding_prefetch,
        )
    else:
        execute_stationary_c_ops(
            a,
            b,
            c,
            ops,
            release_remote_tiles=release_remote_tiles,
        )

    if c_rep_factor > 1:
        # Ensure all ranks finish local compute before any one-sided replica
        # reduction starts writing into origin replica storage.
        if dist.is_initialized():
            dist.barrier()
        dt.reduce_replicas(c)
        # Wait for all source replicas to complete one-sided updates before
        # proceeding to subsequent iterations/uses of C.
        if dist.is_initialized():
            dist.barrier()
    return ops
