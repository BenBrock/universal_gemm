import torch

try:
    from torch._inductor.runtime.triton_compat import triton
    import triton.language as tl
except Exception:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


if triton is not None and tl is not None:
    @triton.jit
    def _accumulate_tile_atomic_add_kernel(
        dst_ptr,
        src_ptr,
        rows,
        cols,
        dst_stride_0,
        dst_stride_1,
        src_stride_0,
        src_stride_1,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask = (offs_m[:, None] < rows) & (offs_n[None, :] < cols)

        dst = dst_ptr + offs_m[:, None] * dst_stride_0 + offs_n[None, :] * dst_stride_1
        src = src_ptr + offs_m[:, None] * src_stride_0 + offs_n[None, :] * src_stride_1
        vals = tl.load(src, mask=mask, other=0)
        tl.atomic_add(dst, vals, mask=mask)


def is_available() -> bool:
    return triton is not None and tl is not None


def launch_atomic_add(dst: torch.Tensor, src: torch.Tensor) -> None:
    if not is_available():
        raise RuntimeError(
            "accumulate_tile requires Triton, but Triton is unavailable in this environment"
        )
    rows, cols = src.shape
    grid = (triton.cdiv(rows, 32), triton.cdiv(cols, 32))
    _accumulate_tile_atomic_add_kernel[grid](
        dst,
        src,
        rows,
        cols,
        dst.stride(0),
        dst.stride(1),
        src.stride(0),
        src.stride(1),
        BLOCK_M=32,
        BLOCK_N=32,
    )
