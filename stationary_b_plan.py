from stationary_c_plan import MultiplyOp
from collections import deque
import time
from typing import TypeVar

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

import dtensor_profile as profile
from tile_bounds import Slice1D, Slice2D, overlapping_tiles, tile_bounds


T = TypeVar("T")


_stationary_b_plan_cache: dict[tuple[object, ...], list[MultiplyOp]] = {}


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


def _placement_key(dt: DTensor) -> tuple[object, ...]:
    placements: list[tuple[str, int | None]] = []
    for placement in dt.placements:
        if isinstance(placement, Shard):
            placements.append(("Shard", int(placement.dim)))
        elif isinstance(placement, Replicate):
            placements.append(("Replicate", None))
        elif isinstance(placement, Partial):
            placements.append(("Partial", None))
        else:
            raise ValueError(f"Unsupported placement in stationary-B cache key: {placement}")
    return tuple(placements)


def _mesh_key(dt: DTensor) -> tuple[object, ...]:
    mesh = dt.device_mesh.mesh
    return (tuple(int(v) for v in mesh.shape), tuple(int(v) for v in mesh.reshape(-1).tolist()))


def _stationary_b_cache_key(
    a: DTensor,
    b: DTensor,
    c: DTensor,
    *,
    local_only: bool,
    rank: int,
) -> tuple[object, ...]:
    return (
        local_only,
        rank,
        tuple(int(v) for v in a.shape),
        tuple(int(v) for v in b.shape),
        tuple(int(v) for v in c.shape),
        _placement_key(a),
        _placement_key(b),
        _placement_key(c),
        _mesh_key(a),
        _mesh_key(b),
        _mesh_key(c),
    )


def clear_stationary_b_plan_cache() -> None:
    _stationary_b_plan_cache.clear()


def _wrapped_replica_m_ranges(
    *,
    m_extent: int,
    rep_factor: int,
    replica_idx: int,
    tile_sum_idx: int,
    m_offsets: list[int],
) -> list[Slice1D]:
    if m_extent <= 0:
        return []
    if not m_offsets:
        m_offsets = [0]

    chunk_size = (m_extent + rep_factor - 1) // rep_factor
    base_start = chunk_size * replica_idx
    base_stop = min(m_extent, chunk_size * (replica_idx + 1))
    base_len = max(base_stop - base_start, 0)
    if base_len == 0:
        return []

    offset = m_offsets[tile_sum_idx % len(m_offsets)] % m_extent
    start = (base_start + offset) % m_extent
    stop = start + base_len
    if stop <= m_extent:
        return [Slice1D(start, stop)]
    return [Slice1D(start, m_extent), Slice1D(0, stop - m_extent)]


def _logical_m_blocks(a: DTensor, c: DTensor) -> list[Slice1D]:
    import dtensor_utils as dt  # Lazy import to avoid circular import at module import time.

    boundaries = {0, int(c.shape[0])}
    a_grid_rows, _ = dt.grid_shape(a)
    c_grid_rows, _ = dt.grid_shape(c)

    for a_row in range(a_grid_rows):
        bounds = tile_bounds(a, (a_row, 0)).rows
        boundaries.add(bounds.start)
        boundaries.add(bounds.stop)

    for c_row in range(c_grid_rows):
        bounds = tile_bounds(c, (c_row, 0)).rows
        boundaries.add(bounds.start)
        boundaries.add(bounds.stop)

    points = sorted(boundaries)
    return [
        Slice1D(points[idx], points[idx + 1])
        for idx in range(len(points) - 1)
        if points[idx] < points[idx + 1]
    ]


def _rotate_group(items: list[T], offset: int) -> list[T]:
    if not items:
        return items
    start = offset % len(items)
    if start == 0:
        return items
    return items[start:] + items[:start]


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
    cache_key = _stationary_b_cache_key(a, b, c, local_only=local_only, rank=rank)
    cached = _stationary_b_plan_cache.get(cache_key)
    if cached is not None:
        return cached

    b_grid_rows, b_grid_cols = dt.grid_shape(b)
    b_rep_factor = dt.replication_factor(b)
    b_replica = dt.my_replica(b)
    m_offsets = [block.start for block in _logical_m_blocks(a, c)]

    ops: list[MultiplyOp] = []

    for k in range(b_grid_rows):
        for j in range(b_grid_cols):
            b_idx = (k, j)
            if local_only and dt.tile_rank(b, b_idx) != rank:
                continue

            global_i_ranges = _wrapped_replica_m_ranges(
                m_extent=int(c.shape[0]),
                rep_factor=b_rep_factor,
                replica_idx=b_replica,
                tile_sum_idx=k + j,
                m_offsets=m_offsets,
            )
            if not global_i_ranges:
                continue

            b_bounds = tile_bounds(b, b_idx)
            grouped_ops: dict[tuple[int, int], list[MultiplyOp]] = {}
            group_order: list[tuple[int, int]] = []

            for global_i_bounds in global_i_ranges:
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

                        key = (m_bounds.start, m_bounds.stop)
                        if key not in grouped_ops:
                            grouped_ops[key] = []
                            group_order.append(key)
                        grouped_ops[key].append(
                            MultiplyOp(
                                a_idx=a_idx,
                                b_idx=b_idx,
                                c_idx=c_idx,
                                a_slice=dt.subtract_offset(global_a, dt.tile_offset(a, a_idx)),
                                b_slice=dt.subtract_offset(global_b, dt.tile_offset(b, b_idx)),
                                c_slice=dt.subtract_offset(global_c, dt.tile_offset(c, c_idx)),
                            )
                        )

            for key in group_order:
                ops.extend(_rotate_group(grouped_ops[key], k + j))
    _stationary_b_plan_cache[cache_key] = ops
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

        begin = time.time()
        a_tile = dt.get_tile(a, op.a_idx)
        end = time.time()
        profile.add_comm_sync(end - begin)
        b_tile = dt.tile(b, op.b_idx)

        a_view = a_tile[*op.a_slice.as_slices()]
        b_view = b_tile[*op.b_slice.as_slices()]
        begin = time.time()
        prod = torch.mm(a_view, b_view)
        if c.device.type == "cuda":
            torch.cuda.current_stream().synchronize()
        end = time.time()
        profile.add_compute(end - begin)

        begin = time.time()
        accum_fut = dt.accumulate_tile(
            c,
            op.c_idx,
            prod,
            slice_=op.c_slice,
        )
        end = time.time()
        profile.add_comm_issue(end - begin)

        begin = time.time()
        accum_fut.wait()
        end = time.time()
        profile.add_comm_sync(end - begin)

        # get_tile uses pooled scratch for remote tiles.
        # Synchronize before releasing/reusing scratch.
        if c.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

        if release_remote_tiles:
            dt.release_tile(a_tile)

    if c.device.type == "cuda":
        torch.cuda.current_stream().synchronize()


def execute_stationary_b_ops_async(
    a: DTensor,
    b: DTensor,
    c: DTensor,
    ops: list[MultiplyOp],
    *,
    release_remote_tiles: bool = True,
    max_outstanding_prefetch: int = 1,
    max_outstanding_accumulates: int = 4,
    num_compute_streams: int = 2,
) -> None:
    """
    Execute a Stationary-B op list with async A-tile prefetch.

    Each op is issued on a stream from a round-robin pool, and both GEMM and
    accumulate for that op run on the same stream. This permits overlap across
    independent op streams.
    """
    import dtensor_utils as dt  # Lazy import to avoid circular import at module import time.

    if max_outstanding_prefetch < 0:
        raise ValueError(
            f"max_outstanding_prefetch must be non-negative, got {max_outstanding_prefetch}"
        )
    if max_outstanding_accumulates <= 0:
        raise ValueError(
            "max_outstanding_accumulates must be positive, "
            f"got {max_outstanding_accumulates}"
        )
    if num_compute_streams <= 0:
        raise ValueError(
            f"num_compute_streams must be positive, got {num_compute_streams}"
        )

    rank = dist.get_rank() if dist.is_initialized() else 0
    if not ops:
        return

    a_tile_requests: dict[tuple[int, int], object] = {}
    a_tiles: dict[tuple[int, int], torch.Tensor] = {}

    # Track requests that consume prefetch budget.
    prefetched_requests: set[tuple[int, int]] = set()

    # Keep outstanding accumulate futures alive with their product buffers.
    outstanding_accumulates: deque[
        tuple[
            object,
            tuple[int, int],
            torch.Tensor,
            tuple[torch.cuda.Event, torch.cuda.Event] | None,
        ]
    ] = deque()

    a_remaining: dict[tuple[int, int], int] = {}
    for op in ops:
        a_remaining[op.a_idx] = a_remaining.get(op.a_idx, 0) + 1

    compute_streams: list[torch.cuda.Stream] = []
    if c.device.type == "cuda":
        compute_streams = [torch.cuda.Stream(device=c.device) for _ in range(num_compute_streams)]

    def _enqueue_a(idx: tuple[int, int], *, prefetch: bool) -> bool:
        if idx in a_tiles or idx in a_tile_requests:
            return False
        begin = time.time()
        a_tile_requests[idx] = dt.get_tile_async(a, idx)
        end = time.time()
        profile.add_comm_issue(end - begin)
        if prefetch:
            prefetched_requests.add(idx)
        return True

    def _materialize_a(idx: tuple[int, int]) -> torch.Tensor:
        future = a_tile_requests.pop(idx, None)
        if future is not None:
            begin = time.time()
            a_tile = future.get()
            end = time.time()
            profile.add_comm_sync(end - begin)
            # Remote get_tile_async uses pooled scratch. Clone so cached tiles are
            # independent from scratch reuse while later GEMMs are still in flight.
            if dt.tile_rank(a, idx) != rank:
                a_tile = a_tile.clone()
            a_tiles[idx] = a_tile
        else:
            a_tile = a_tiles.get(idx)
            if a_tile is None:
                begin = time.time()
                future = dt.get_tile_async(a, idx)
                end = time.time()
                profile.add_comm_issue(end - begin)
                begin = time.time()
                a_tile = future.get()
                end = time.time()
                profile.add_comm_sync(end - begin)
                if dt.tile_rank(a, idx) != rank:
                    a_tile = a_tile.clone()
                a_tiles[idx] = a_tile
        prefetched_requests.discard(idx)
        return a_tiles[idx]

    def _prefetch_next(next_op: MultiplyOp | None) -> None:
        if next_op is None or max_outstanding_prefetch == 0:
            return
        if len(prefetched_requests) >= max_outstanding_prefetch:
            return
        _enqueue_a(next_op.a_idx, prefetch=True)

    def _evict_a_if_dead(a_idx: tuple[int, int]) -> None:
        if a_remaining[a_idx] != 0:
            return
        for _, in_flight_a_idx, _, _ in outstanding_accumulates:
            if in_flight_a_idx == a_idx:
                return
        tile = a_tiles.pop(a_idx, None)
        if release_remote_tiles and tile is not None:
            dt.release_tile(tile)

    for op_idx, op in enumerate(ops):
        _validate_op_shapes(op)

        owner = dt.tile_rank(b, op.b_idx)
        if owner != rank:
            raise RuntimeError(
                f"Stationary-B async execution expected local b_idx={op.b_idx}, "
                f"owner={owner}, rank={rank}"
            )

        a_tile = _materialize_a(op.a_idx)
        next_op = ops[op_idx + 1] if op_idx + 1 < len(ops) else None
        _prefetch_next(next_op)

        b_tile = dt.tile(b, op.b_idx)

        a_view = a_tile[*op.a_slice.as_slices()]
        b_view = b_tile[*op.b_slice.as_slices()]

        if not compute_streams:
            begin = time.time()
            prod = torch.mm(a_view, b_view)
            end = time.time()
            profile.add_compute(end - begin)

            begin = time.time()
            accum_fut = dt.accumulate_tile(
                c,
                op.c_idx,
                prod,
                slice_=op.c_slice,
            )
            end = time.time()
            profile.add_comm_issue(end - begin)
            compute_events = None
        else:
            op_stream = compute_streams[op_idx % len(compute_streams)]
            begin_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(op_stream):
                begin_evt.record()
                prod = torch.mm(a_view, b_view)
                end_evt.record()
                begin = time.time()
                accum_fut = dt.accumulate_tile(
                    c,
                    op.c_idx,
                    prod,
                    slice_=op.c_slice,
                )
                end = time.time()
                profile.add_comm_issue(end - begin)
            compute_events = (begin_evt, end_evt)

        outstanding_accumulates.append((accum_fut, op.a_idx, prod, compute_events))

        a_remaining[op.a_idx] -= 1
        _evict_a_if_dead(op.a_idx)

        # Bound in-flight accumulate buffers.
        while len(outstanding_accumulates) > max_outstanding_accumulates:
            oldest_fut, oldest_a_idx, _oldest_prod, oldest_compute_events = (
                outstanding_accumulates.popleft()
            )
            begin = time.time()
            oldest_fut.wait()
            end = time.time()
            wait_duration = end - begin
            if oldest_compute_events is not None:
                begin_evt, end_evt = oldest_compute_events
                compute_duration = begin_evt.elapsed_time(end_evt) / 1000.0
                profile.add_compute(compute_duration)
                profile.add_comm_sync(max(wait_duration - compute_duration, 0.0))
            else:
                profile.add_comm_sync(wait_duration)
            _evict_a_if_dead(oldest_a_idx)

    # Drain all in-flight accumulates.
    while outstanding_accumulates:
        fut, a_idx, _prod, compute_events = outstanding_accumulates.popleft()
        begin = time.time()
        fut.wait()
        end = time.time()
        wait_duration = end - begin
        if compute_events is not None:
            begin_evt, end_evt = compute_events
            compute_duration = begin_evt.elapsed_time(end_evt) / 1000.0
            profile.add_compute(compute_duration)
            profile.add_comm_sync(max(wait_duration - compute_duration, 0.0))
        else:
            profile.add_comm_sync(wait_duration)
        _evict_a_if_dead(a_idx)

    # Drain any prefetched requests that were never consumed.
    for idx, future in a_tile_requests.items():
        begin = time.time()
        tile = future.get()
        end = time.time()
        profile.add_comm_sync(end - begin)
        if release_remote_tiles:
            dt.release_tile(tile)

    if c.device.type == "cuda":
        for stream in compute_streams:
            stream.synchronize()
        torch.cuda.current_stream().synchronize()


def execute_stationary_b(
    a: DTensor,
    b: DTensor,
    c: DTensor,
    *,
    local_only: bool = True,
    release_remote_tiles: bool = True,
    use_async: bool | None = None,
    max_outstanding_prefetch: int = 1,
    max_outstanding_accumulates: int = 4,
    num_compute_streams: int = 2,
) -> list[MultiplyOp]:
    """
    Build and execute a Stationary-B plan.
    Returns the generated op list for debugging/printing.
    """
    import dtensor_utils as dt  # Lazy import to avoid circular import at module import time.
    import os

    c_rep_factor = dt.replication_factor(c)
    if c_rep_factor > 1 and dt.my_replica(c) != 0:
        # We reduce replica-local C chunks into origin replica 0.
        # Clear non-origin local C each run to avoid stale carry-over.
        c.to_local().zero_()

    ops = build_stationary_b_ops(a, b, c, local_only=local_only)
    if use_async is None:
        use_async = True
    if use_async:
        execute_stationary_b_ops_async(
            a,
            b,
            c,
            ops,
            release_remote_tiles=release_remote_tiles,
            max_outstanding_prefetch=max_outstanding_prefetch,
            max_outstanding_accumulates=max_outstanding_accumulates,
            num_compute_streams=num_compute_streams,
        )
    else:
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
