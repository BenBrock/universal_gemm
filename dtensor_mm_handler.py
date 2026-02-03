import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
import dtensor_utils as dt


aten = torch.ops.aten

def _row_partitioned_matmul(a: DTensor, b: DTensor, c: DTensor):
    # a, b, c have already been verified at this point.

    for i in range(dt.grid_shape(a)[0]):
        if dist.get_rank() == dt.tile_rank(a, (i, 0)):
            a_tile = dt.tile(a, (i, 0))
            c_tile = dt.tile(c, (i, 0))

            for k_ in range(dt.grid_shape(b)[0]):
                k = (k_ + i) % dt.grid_shape(b)[0]
                b_tile = dt.get_tile(b, (k, 0))

                tile_shape = dt.tile_shape(b)

                a_view = a_tile[:,k*tile_shape[0]:(k+1)*tile_shape[0]]

                torch.addmm(c_tile, a_view, b_tile, beta=1.0, alpha=1.0, out=c_tile)


def _mm_out_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    a, b = args[0], args[1]
    c = kwargs.get("out", args[2] if len(args) > 2 else None)
    if not (isinstance(a, DTensor) and isinstance(b, DTensor) and isinstance(c, DTensor)):
        raise RuntimeError("aten.mm.out handler expects a, b, and out to be DTensors")
    if not (a.ndim == 2 and b.ndim == 2 and c.ndim == 2):
        raise RuntimeError("aten.mm.out handler expects a, b, and out to be rank-2 DTensors")

    for name, t in (("a", a), ("b", b), ("out", c)):
        if not (
            t.device_mesh.ndim == 1
            and len(t.placements) == 1
            and isinstance(t.placements[0], Shard)
            and t.placements[0].dim == 0
        ):
            raise RuntimeError(
                f"NotImplemented: aten.mm.out handler expects {name} to be row-sharded on a 1D mesh"
            )

    if a.shape[1] != b.shape[0] or c.shape[0] != a.shape[0] or c.shape[1] != b.shape[1]:
        raise RuntimeError(
            f"aten.mm.out handler shape mismatch: a.shape={tuple(a.shape)}, "
            f"b.shape={tuple(b.shape)}, c.shape={tuple(c.shape)}")

    _row_partitioned_matmul(a, b, c)


_CUSTOM_OPS = {
    aten.mm.out: _mm_out_handler,
}


def enable() -> None:
    DTensor._op_dispatcher._custom_op_handlers.update(_CUSTOM_OPS)


def disable() -> None:
    for op in _CUSTOM_OPS:
        DTensor._op_dispatcher._custom_op_handlers.pop(op, None)
