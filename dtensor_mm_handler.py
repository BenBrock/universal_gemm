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


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _maybe_print_stats(op_call, op_info) -> None:
    if _rank() != 0:
        return
    try:
        schema = op_info.schema
        args = op_info.local_args
        lhs = args[0]
        rhs = args[1]
        lhs_spec = None
        rhs_spec = None
        if schema is not None:
            specs = schema.args_spec
            if len(specs) >= 2:
                lhs_spec, rhs_spec = specs[0], specs[1]
        lhs_placements = lhs_spec.placements if lhs_spec is not None else None
        rhs_placements = rhs_spec.placements if rhs_spec is not None else None
        out_spec = (
            op_info.output_sharding.output_spec
            if op_info.output_sharding is not None
            else None
        )
        out_placements = out_spec.placements if hasattr(out_spec, "placements") else None
        print(
            f"[dtensor_mm_handler] {op_call} "
            f"lhs={tuple(lhs.shape)} rhs={tuple(rhs.shape)} "
            f"dtype={lhs.dtype} "
            f"lhs_placements={lhs_placements} "
            f"rhs_placements={rhs_placements} "
            f"out_placements={out_placements}"
        )
    except Exception as exc:
        print(f"[dtensor_mm_handler] stats unavailable: {exc}")


def _mm_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    schema = op_info.schema
    args = op_info.local_args
    specs = schema.args_spec
    if dist.get_rank() == 0:
        print(f'op_info: {op_info}')
        print(f'op_info.schema: {op_info.schema}')
        print(f'op_info.local_args: {op_info.local_args}')
        print(f'schema.args_spec: {schema.args_spec}')

    return
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"

    _maybe_print_stats(op_call, op_info)

    schema = op_info.schema
    if schema is not None:
        specs = schema.args_spec
        if len(specs) >= 2:
            lhs_spec, rhs_spec = specs[0], specs[1]
            if (
                lhs_spec.mesh.ndim == 1
                and rhs_spec.mesh.ndim == 1
                and len(lhs_spec.placements) == 1
                and len(rhs_spec.placements) == 1
                and isinstance(lhs_spec.placements[0], Shard)
                and isinstance(rhs_spec.placements[0], Shard)
                and lhs_spec.placements[0].dim == 0
                and rhs_spec.placements[0].dim == 0
            ):
                # Path A: both inputs are row-sharded on a 1D mesh.
                # For now, intentionally do nothing (return zeros) to force
                # the validation check to fail.
                out_spec = output_sharding.output_spec
                assert isinstance(out_spec, type(lhs_spec))
                assert out_spec.tensor_meta is not None
                local_shape, _ = compute_local_shape_and_global_offset(
                    out_spec.tensor_meta.shape,
                    out_spec.mesh,
                    out_spec.placements,
                )
                device = op_info.local_args[0].device
                local_results = torch.zeros(
                    local_shape,
                    dtype=out_spec.tensor_meta.dtype,
                    device=device,
                )
                return DTensor._op_dispatcher.wrap(
                    local_results, output_sharding.output_spec
                )

    # Path B: default DTensor behavior (redistribute if needed, then run local op).
    print('Default path!')
    if output_sharding.needs_redistribute:
        assert output_sharding.redistribute_schema is not None
        DTensor._op_dispatcher.redistribute_local_args(
            op_info,
            output_sharding.redistribute_schema,
            output_sharding.use_val_from_redistribute_schema,
        )

    local_results = op_call(*op_info.local_args, **op_info.local_kwargs)
    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


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
    aten.mm.default: _mm_handler,
    aten.mm.out: _mm_out_handler,
}


def enable() -> None:
    DTensor._op_dispatcher._custom_op_handlers.update(_CUSTOM_OPS)


def disable() -> None:
    for op in _CUSTOM_OPS:
        DTensor._op_dispatcher._custom_op_handlers.pop(op, None)
