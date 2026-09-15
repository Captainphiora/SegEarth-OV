# Scaled Dot Product Attention Contract

## Identity

- Target API: `torch.nn.functional.scaled_dot_product_attention`.
- Missing path: 310B has no usable flash/efficient SDPA kernel for this project.
- Compatibility evidence: `npu_compat.py:46-82` decomposes SDPA into matmul,
  masking, softmax, dropout, and matmul.

## Required Semantics

Implement `softmax((Q @ K^T) * scale + mask) @ V` with PyTorch-compatible shape,
mask broadcasting, causal behavior, output dtype/shape, and optional GQA head
mapping. Boolean masks retain `True` elements; additive masks are added before
softmax. `scale=None` means `1/sqrt(E)`.

The first inference scope may require `dropout_p == 0`; any other value must fail
clearly until an owned random/dropout contract is implemented. Do not silently
ignore dropout. Do not call PyTorch SDPA, FlashAttention, or a vendor whole SDPA
operator from the candidate path.

## Minimum Validation Matrix

- FP16 rank-4 Q/K/V with self-attention and cross-attention (`L != S`).
- Batch and head broadcasting boundaries; non-multiple sequence/tail lengths.
- No mask, boolean mask, additive mask, and `is_causal=True` separately.
- Default and explicit scale; GQA disabled and one valid enabled case if claimed.
- Fully masked rows, finite large logits, and tolerance recorded against an
  untimed FP32 CPU golden.
- At least one `open_clip` attention call with the global patch disabled.
