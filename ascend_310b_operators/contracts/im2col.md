# Im2col / Unfold Contract

## Identity

- Target API: `torch.nn.functional.unfold` for rank-4 NCHW tensors.
- Missing path: `aclnnIm2col` is unavailable in the observed 310B runtime.
- Compatibility evidence: `npu_compat.py:145-182` uses padding, index selection,
  permutation, and reshape.

## Required Semantics

For input `[N,C,H,W]`, return `[N, C*Kh*Kw, L]` in PyTorch Unfold ordering, with
`L = Oh*Ow` computed from kernel, dilation, padding, and stride. Preserve kernel
element ordering, channel ordering, dtype, and zero-padding semantics.

Do not call `F.unfold`, Im2col, or a CPU extraction path as the whole candidate
implementation. Workspace and index storage must be bounded from runtime shapes.

## Minimum Validation Matrix

- FP16 and any additionally claimed dtype; batch/channel greater than one.
- Scalar and pair forms for kernel, dilation, padding, and stride.
- `1x1`, square, and non-square kernels; overlapping and non-overlapping windows.
- Zero and nonzero padding, dilation greater than one, non-square inputs, and tail
  dimensions not aligned to vector/tile widths.
- Exact output shape/order checks and value comparison against untimed CPU Unfold.
- At least one real project path that exercises Unfold with the patch disabled;
  if no current call site exists, record that integration gate as not applicable.
