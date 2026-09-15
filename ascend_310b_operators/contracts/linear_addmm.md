# Linear / Addmm Contract

## Identity

- Project API: `torch.nn.functional.linear`.
- Missing path: `aclnnAddmm` selects an unavailable `MatMulV2` kernel on 310B for
  the observed FP16/internal-format path.
- Compatibility evidence: `npu_compat.py:85-101` uses matmul followed by bias add.

## Required Semantics

For Linear, compute `input @ weight.T + bias` with input `[..., K]`, weight
`[N, K]`, optional bias `[N]`, and output `[..., N]`. If Addmm is claimed as a
separate API, also implement its full `beta * input + alpha * (mat1 @ mat2)`
contract; Linear-only evidence must not be labeled Addmm-complete.

FP32 on 310B Cube may require an explicit owned cast-compute-cast policy. Record
the precision contract and tolerance; do not silently downcast unsupported input
dtypes. Do not call `F.linear`, `torch.addmm`, or vendor Addmm/Linear as the whole
candidate implementation.

## Minimum Validation Matrix

- FP16 2-D and rank-3/batched inputs; bias present and absent.
- `M`, `N`, and `K` values both aligned and non-aligned to tile boundaries.
- Small, tail-heavy, and project-representative `open_clip/transformer.py` shapes.
- Non-contiguous input/weight is either supported correctly or rejected clearly.
- FP32 only when its owned precision path and tolerance are explicitly proven.
- Compare output shape, dtype, and values against an untimed CPU golden.
