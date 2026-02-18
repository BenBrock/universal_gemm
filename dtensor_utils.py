import torch
import torch.distributed as dist
import nvshmem.core as nvshmem
from cuda.core.experimental import Device
from cuda.core.experimental._stream import Stream

from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
import accumulate_kernels
from dtensor_scratch import NvshmemHeap
from tile_bounds import Slice1D, Slice2D, overlapping_tiles, tile_bounds
from stationary_b_plan import (
    build_stationary_b_ops,
    execute_stationary_b,
    execute_stationary_b_ops,
    format_stationary_b_op,
    format_stationary_b_plan,
)
from stationary_c_plan import (
    MultiplyOp,
    build_stationary_c_ops,
    execute_stationary_c,
    execute_stationary_c_ops,
    format_op,
    format_plan,
)

_tile_heap: NvshmemHeap | None = None
_tile_heap_dtype: torch.dtype | None = None
_tile_heap_device: torch.device | None = None
_get_tile_async_raw_nvshmem_stream: Stream | None = None


def _get_async_nvshmem_stream() -> Stream:
    global _get_tile_async_raw_nvshmem_stream
    if _get_tile_async_raw_nvshmem_stream is None:
        device_id = torch.cuda.current_device()
        _get_tile_async_raw_nvshmem_stream = Device(device_id).create_stream()
    return _get_tile_async_raw_nvshmem_stream


def _get_tile_heap(dt: DTensor) -> NvshmemHeap:
    if _tile_heap is None:
        raise RuntimeError(
            "get_tile requires init_scratch(...) to be called collectively before use."
        )
    if _tile_heap_dtype != dt.dtype or _tile_heap_device != dt.device:
        raise RuntimeError(
            "get_tile requires init_scratch(...) to be called for this DTensor dtype/device "
            "before use."
        )
    return _tile_heap


def init_scratch(
    num_elements: int,
    dtype: torch.dtype,
) -> None:
    """
    Allocate a simple NVSHMEM heap for temporary tiles (collective).
    """
    global _tile_heap, _tile_heap_dtype, _tile_heap_device
    device = torch.device("cuda", torch.cuda.current_device())
    if (
        _tile_heap is None
        or _tile_heap_dtype != dtype
        or _tile_heap_device != device
        or _tile_heap.capacity < num_elements
    ):
        if _tile_heap is not None:
            _tile_heap.close()
        _tile_heap = NvshmemHeap(num_elements, dtype)
        _tile_heap_dtype = dtype
        _tile_heap_device = device
    _tile_heap.reset()


def free_get_tile_scratch(dt: DTensor | None = None) -> None:
    """
    Free the NVSHMEM heap (collective).
    """
    global _tile_heap, _tile_heap_dtype, _tile_heap_device
    if _tile_heap is None:
        return
    if dt is not None and (dt.dtype != _tile_heap_dtype or dt.device != _tile_heap_device):
        return
    _tile_heap.close()
    _tile_heap = None
    _tile_heap_dtype = None
    _tile_heap_device = None


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


def matrix_numel(dt: DTensor) -> int:
    """
    Return total element count for a rank-2 DTensor.
    """
    if dt.ndim != 2:
        raise ValueError(f"matrix_numel expects a 2D DTensor, got ndim={dt.ndim}")
    return int(dt.shape[0]) * int(dt.shape[1])


def replication_factor(dt: DTensor) -> int:
    """
    Return the number of logical replicas of a DTensor.

    Replication factor is the product of mesh sizes for all Replicate placements.
    Shard and Partial placements do not increase the replica count.
    """
    factor = 1
    mesh = dt.device_mesh
    for mesh_dim, placement in enumerate(dt.placements):
        if isinstance(placement, Replicate):
            factor *= mesh.size(mesh_dim)
        elif isinstance(placement, (Shard, Partial)):
            continue
        else:
            raise ValueError(f"replication_factor does not support placement {placement}")
    return factor


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


def _flatten_index(coords: list[int], sizes: list[int]) -> int:
    if len(coords) != len(sizes):
        raise ValueError(f"coords/sizes length mismatch: {len(coords)} != {len(sizes)}")
    index = 0
    for c, size in zip(coords, sizes):
        if c < 0 or c >= size:
            raise ValueError(f"coordinate {c} out of range for size {size}")
        index = index * size + c
    return index


def _replicate_mesh_dims(dt: DTensor) -> list[int]:
    return [
        mesh_dim
        for mesh_dim, placement in enumerate(dt.placements)
        if isinstance(placement, Replicate)
    ]


def my_replica(dt: DTensor) -> int:
    """
    Return the replica index associated with the calling process.

    If dt is not replicated, returns 0.
    """
    rep_dims = _replicate_mesh_dims(dt)
    if not rep_dims:
        return 0

    coord = dt.device_mesh.get_coordinate()
    if coord is None:
        raise RuntimeError("Current rank is not part of the DTensor device mesh")

    rep_sizes = [dt.device_mesh.size(d) for d in rep_dims]
    rep_coords = [coord[d] for d in rep_dims]
    return _flatten_index(rep_coords, rep_sizes)


def tile_rank(dt: DTensor, coord: tuple[int, int], replica: int | None = None) -> int:
    placements = dt.placements
    mesh = dt.device_mesh

    rep_dims = _replicate_mesh_dims(dt)
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

    full_coord = [0] * mesh.ndim
    if replica is None:
        replica = my_replica(dt)

    rep_sizes = [mesh.size(d) for d in rep_dims]
    rep_factor = 1
    for s in rep_sizes:
        rep_factor *= s
    if replica < 0 or replica >= rep_factor:
        raise ValueError(
            f"replica {replica} out of range for replication_factor {rep_factor}"
        )

    if rep_dims:
        rep_coords = _unflatten_index(replica, rep_sizes)
        for d, c in zip(rep_dims, rep_coords):
            full_coord[d] = c

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
    
    return int(mesh.mesh[tuple(full_coord)].item())


def intersect_1d(a: Slice1D, b: Slice1D) -> Slice1D:
    start = max(a.start, b.start)
    stop = min(a.stop, b.stop)
    if stop < start:
        stop = start
    return Slice1D(start, stop)


def intersect_2d(a: Slice2D, b: Slice2D) -> Slice2D:
    return Slice2D(intersect_1d(a.rows, b.rows), intersect_1d(a.cols, b.cols))


def tile_offset(dt: DTensor, coord: tuple[int, int]) -> tuple[int, int]:
    bounds = tile_bounds(dt, coord)
    return (bounds.rows.start, bounds.cols.start)


def subtract_offset(region: Slice2D, offset: tuple[int, int]) -> Slice2D:
    return Slice2D(
        rows=Slice1D(region.rows.start - offset[0], region.rows.stop - offset[0]),
        cols=Slice1D(region.cols.start - offset[1], region.cols.stop - offset[1]),
    )


def get_tile(dt: DTensor, coord: tuple[int, int], replica: int | None = None) -> torch.Tensor:
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
    owner_rank = tile_rank(dt, coord, replica=replica)
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Fast path for local ownership: avoid scratch allocation and NVSHMEM get.
    if owner_rank == rank:
        rows, cols = actual_shape
        local = dt.to_local()
        return local[:rows, :cols]

    heap = _get_tile_heap(dt)
    scratch = heap.alloc(max_shape)
    src = dt.nvshmem_base().view(max_shape)
    stream = torch.cuda.current_stream()
    nvshmem.get(scratch._nvshmem_buf, src, remote_pe=owner_rank, stream=stream)
    nvshmem.quiet(stream)
    stream.synchronize()
    rows, cols = actual_shape
    return scratch[:rows, :cols]


def tile(dt: DTensor, coord: tuple[int, int], replica: int | None = None) -> torch.Tensor:
    """
    Return a local view of the tile if owned by the calling rank.
    Raises if the tile is remote.
    """
    if dt.ndim != 2:
        raise ValueError(f"tile expects a 2D DTensor, got ndim={dt.ndim}")

    owner_rank = tile_rank(dt, coord, replica=replica)
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
    return


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


class _AccumulateFuture:
    def __init__(self, event: torch.cuda.Event) -> None:
        self._event = event

    def wait(self) -> None:
        self._event.synchronize()


class _ReduceReplicasFuture:
    def __init__(self, futures: list[_AccumulateFuture]) -> None:
        self._futures = futures

    def wait(self) -> None:
        for fut in self._futures:
            fut.wait()


def get_tile_async(
    dt: DTensor, coord: tuple[int, int], replica: int | None = None
) -> _TileFuture:
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
    owner_rank = tile_rank(dt, coord, replica=replica)

    heap = _get_tile_heap(dt)
    scratch = heap.alloc(max_shape)
    nvshmem_stream = _get_async_nvshmem_stream()
    src = dt.nvshmem_base().view(max_shape)
    nvshmem.get(scratch._nvshmem_buf, src, remote_pe=owner_rank, stream=nvshmem_stream)
    return _TileFuture(scratch, nvshmem_stream, rows, cols)


def accumulate_tile(
    dt: DTensor,
    coord: tuple[int, int],
    view: torch.Tensor,
    *,
    slice_: Slice2D | None = None,
    replica: int | None = None,
) -> _AccumulateFuture:
    """
    Asynchronously accumulate `view` into a local/remote DTensor tile slice.

    This API is one-sided and non-collective:
      - local tile owner path updates local memory directly
      - remote tile owner path maps peer memory with nvshmem.get_peer_tensor(...)
        and uses atomic add on that mapped pointer
    """
    if dt.ndim != 2:
        raise ValueError(f"accumulate_tile expects a 2D DTensor, got ndim={dt.ndim}")
    if view.ndim != 2:
        raise ValueError(f"accumulate_tile expects a rank-2 view, got ndim={view.ndim}")
    if dt.device.type != "cuda" or view.device.type != "cuda":
        raise RuntimeError("accumulate_tile currently supports CUDA tensors only")
    if view.dtype != dt.dtype:
        raise RuntimeError(
            f"accumulate_tile dtype mismatch: view={view.dtype}, dt={dt.dtype}"
        )
    if view.dtype != torch.float32:
        raise RuntimeError(
            f"accumulate_tile currently supports torch.float32 only, got {view.dtype}"
        )
    if not accumulate_kernels.is_available():
        raise RuntimeError(
            "accumulate_tile requires Triton, but Triton is unavailable in this environment"
        )

    grid_rows, grid_cols = grid_shape(dt)
    row_idx, col_idx = coord
    if not (0 <= row_idx < grid_rows and 0 <= col_idx < grid_cols):
        raise ValueError(
            f"tile coord {coord} out of range for grid {(grid_rows, grid_cols)}"
        )

    tile_rows, tile_cols = tile_shape(dt, coord)
    if slice_ is None:
        slice_ = Slice2D(Slice1D(0, tile_rows), Slice1D(0, tile_cols))

    if slice_.rows.stop > tile_rows or slice_.cols.stop > tile_cols:
        raise ValueError(
            f"slice {slice_} out of bounds for tile shape {(tile_rows, tile_cols)}"
        )
    if view.shape != slice_.shape:
        raise ValueError(
            f"view shape {tuple(view.shape)} does not match slice shape {slice_.shape}"
        )

    owner_rank = tile_rank(dt, coord, replica=replica)
    rank = dist.get_rank() if dist.is_initialized() else 0

    max_shape = tile_shape(dt)
    raw_base = dt.nvshmem_base()
    is_nvshmem_backed = bool(getattr(raw_base, "_nvshmem_alloc", False))
    local_base = raw_base.view(max_shape)
    if owner_rank == rank:
        dst_base = local_base
    else:
        if not is_nvshmem_backed:
            raise RuntimeError(
                "accumulate_tile remote updates require NVSHMEM-backed DTensor storage "
                "(set TORCH_DTENSOR_USE_NVSHMEM=1 before constructing DTensors)"
            )
        try:
            dst_base = nvshmem.get_peer_tensor(local_base, owner_rank).view(max_shape)
        except Exception as exc:
            raise RuntimeError(
                "accumulate_tile failed to map remote tile memory with "
                "nvshmem.get_peer_tensor; this path requires single-node GPU P2P access"
            ) from exc

    dst_view = dst_base[*slice_.as_slices()]
    if dst_view.shape != view.shape:
        raise RuntimeError(
            f"internal accumulate_tile shape mismatch: dst={tuple(dst_view.shape)} "
            f"view={tuple(view.shape)}"
        )

    accumulate_kernels.launch_atomic_add(dst_view, view)

    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream())
    return _AccumulateFuture(event)


def reduce_replicas_async(
    dt: DTensor,
    *,
    origin_replica: int = 0,
) -> _ReduceReplicasFuture:
    """
    Reduce all replicas into `origin_replica` using one-sided accumulate_tile updates.

    This helper is collective-by-convention: all ranks in the DTensor mesh should
    call it. Each non-origin replica owner pushes its local tile to the
    corresponding tile in `origin_replica`.
    """
    if dt.ndim != 2:
        raise ValueError(f"reduce_replicas_async expects a 2D DTensor, got ndim={dt.ndim}")

    rep_factor = replication_factor(dt)
    if rep_factor <= 1:
        return _ReduceReplicasFuture([])
    if origin_replica < 0 or origin_replica >= rep_factor:
        raise ValueError(
            f"origin_replica {origin_replica} out of range for replication_factor {rep_factor}"
        )

    rank = dist.get_rank() if dist.is_initialized() else 0
    rows, cols = grid_shape(dt)
    futures: list[_AccumulateFuture] = []

    for replica_idx in range(rep_factor):
        if replica_idx == origin_replica:
            continue
        for i in range(rows):
            for j in range(cols):
                coord = (i, j)
                owner = tile_rank(dt, coord, replica=replica_idx)
                if owner != rank:
                    continue
                src_tile = tile(dt, coord, replica=replica_idx)
                futures.append(
                    accumulate_tile(
                        dt,
                        coord,
                        src_tile,
                        replica=origin_replica,
                    )
                )
    return _ReduceReplicasFuture(futures)


def reduce_replicas(
    dt: DTensor,
    *,
    origin_replica: int = 0,
) -> None:
    """
    Synchronous wrapper over reduce_replicas_async.
    """
    reduce_replicas_async(dt, origin_replica=origin_replica).wait()
