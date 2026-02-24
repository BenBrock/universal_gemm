import torch
import torch.distributed as dist
import time
from typing import Literal
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
import dtensor_utils as dt


aten = torch.ops.aten

comm_issue = 0
comm_sync = 0
compute = 0
StationaryMethod = Literal["auto", "stationary_c", "stationary_b"]
_stationary_method: StationaryMethod = "auto"


def print_stats():
    global comm_issue
    global comm_sync
    global compute
    total = comm_issue + comm_sync + compute
    print(f'comm_issue: {comm_issue}, comm_sync: {comm_sync}, compute: {compute}')
    if total > 0:
        print(f'comm_issue: {100*(comm_issue/total)}%, comm_sync: {100*(comm_sync/total)}%, compute: {100*(compute/total)}')

def _row_partitioned_matmul_async(a: DTensor, b: DTensor, c: DTensor):
    # a, b, c have already been verified at this point.
    global comm_issue
    global comm_sync
    global compute

    for i in range(dt.grid_shape(a)[0]):
        if dist.get_rank() == dt.tile_rank(a, (i, 0)):
            a_tile = dt.tile(a, (i, 0))
            c_tile = dt.tile(c, (i, 0))

            begin = time.time()
            b_f = dt.get_tile_async(b, (i, 0))
            end = time.time()
            comm_issue += end - begin

            for k_ in range(dt.grid_shape(b)[0]):
                k = (k_ + i) % dt.grid_shape(b)[0]

                begin = time.time()
                b_tile = b_f.get()
                end = time.time()
                comm_sync += end - begin

                if k_ + 1 < dt.grid_shape(b)[0]:
                    begin = time.time()
                    b_f = dt.get_tile_async(b, ((k + 1) % dt.grid_shape(b)[0], 0))
                    end = time.time()
                    comm_issue += end - begin

                tile_shape = dt.tile_shape(b)

                a_view = a_tile[:,k*tile_shape[0]:(k+1)*tile_shape[0]]

                begin = time.time()
                torch.addmm(c_tile, a_view, b_tile, out=c_tile)
                torch.cuda.current_stream().synchronize()
                end = time.time()
                compute += end - begin
                dt.release_tile(b_tile)

def _row_partitioned_matmul(a: DTensor, b: DTensor, c: DTensor):
    # a, b, c have already been verified at this point.
    global comm_issue
    global comm_sync
    global compute

    for i in range(dt.grid_shape(a)[0]):
        if dist.get_rank() == dt.tile_rank(a, (i, 0)):
            a_tile = dt.tile(a, (i, 0))
            c_tile = dt.tile(c, (i, 0))

            for k_ in range(dt.grid_shape(b)[0]):
                k = (k_ + i) % dt.grid_shape(b)[0]

                begin = time.time()
                b_tile = dt.get_tile(b, (k,0))
                end = time.time()
                comm_sync += end - begin

                tile_shape = dt.tile_shape(b)

                a_view = a_tile[:,k*tile_shape[0]:(k+1)*tile_shape[0]]

                begin = time.time()
                torch.addmm(c_tile, a_view, b_tile, out=c_tile)
                torch.cuda.current_stream().synchronize()
                end = time.time()
                compute += end - begin


def _addmm_out_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    # We are computing c = ab + e
    e, a, b = args[0], args[1], args[2]
    c = kwargs.get("out")
    if c is not e:
        raise RuntimeError(
            "NotImplemented: aten.addmm.out handler requires input and output to alias"
        )
    if not (isinstance(a, DTensor) and isinstance(b, DTensor) and isinstance(c, DTensor)):
        raise RuntimeError("aten.addmm.out handler expects a, b, and out to be DTensors")
    if not (a.ndim == 2 and b.ndim == 2 and c.ndim == 2):
        raise RuntimeError("aten.addmm.out handler expects a, b, and out to be rank-2 DTensors")

    if a.shape[1] != b.shape[0] or c.shape[0] != a.shape[0] or c.shape[1] != b.shape[1]:
        raise RuntimeError(
            f"aten.addmm.out handler shape mismatch: a.shape={tuple(a.shape)}, "
            f"b.shape={tuple(b.shape)}, c.shape={tuple(c.shape)}")

    method = _stationary_method
    if method == "stationary_b":
        dt.execute_stationary_b(a, b, c)
    elif method == "stationary_c":
        dt.execute_stationary_c(a, b, c)
    else:
        # Auto-select Stationary-C unless B is larger than C.
        if dt.matrix_numel(b) > dt.matrix_numel(c):
            dt.execute_stationary_b(a, b, c)
        else:
            dt.execute_stationary_c(a, b, c)


_CUSTOM_OPS = {
    aten.addmm.out: _addmm_out_handler,
}


def enable(stationary_method: StationaryMethod = "auto") -> None:
    global _stationary_method
    if stationary_method not in {"auto", "stationary_c", "stationary_b"}:
        raise ValueError(
            "stationary_method must be one of: auto, stationary_c, stationary_b"
        )
    _stationary_method = stationary_method
    DTensor._op_dispatcher._custom_op_handlers.update(_CUSTOM_OPS)


def disable() -> None:
    global _stationary_method
    _stationary_method = "auto"
    for op in _CUSTOM_OPS:
        DTensor._op_dispatcher._custom_op_handlers.pop(op, None)
