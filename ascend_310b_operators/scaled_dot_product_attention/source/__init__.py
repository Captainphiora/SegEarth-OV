"""Owned Ascend 310B scaled-dot-product attention composition."""

import math

import torch
import torch_npu  # noqa: F401


def _require_npu_tensor(tensor, name):
    if not getattr(tensor, "is_npu", False):
        raise ValueError(f"{name} must be an NPU tensor")


def scaled_dot_product_attention_310b(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """Inference SDPA built only from supported NPU tensor primitives."""
    for tensor, name in ((query, "query"), (key, "key"), (value, "value")):
        _require_npu_tensor(tensor, name)
    if query.dtype != torch.float16 or key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("Ascend 310B SDPA currently supports matching float16 Q/K/V only")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be rank-4 tensors")
    if key.shape[-2] != value.shape[-2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("incompatible SDPA sequence or embedding dimensions")
    if dropout_p != 0.0:
        raise ValueError("dropout_p must be 0.0 for the inference-only 310B operator")
    if is_causal and attn_mask is not None:
        raise ValueError("attn_mask and is_causal cannot be enabled together")

    if enable_gqa:
        query_heads = query.shape[-3]
        key_heads = key.shape[-3]
        if key_heads != value.shape[-3] or query_heads % key_heads != 0:
            raise ValueError("GQA requires matching K/V heads that divide query heads")
        repeats = query_heads // key_heads
        if repeats != 1:
            key = key.repeat_interleave(repeats, dim=-3)
            value = value.repeat_interleave(repeats, dim=-3)

    scale_factor = 1.0 / math.sqrt(query.shape[-1]) if scale is None else float(scale)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    allowed_mask = None
    if is_causal:
        rows = torch.arange(query.shape[-2], device=query.device).reshape(-1, 1)
        cols = torch.arange(key.shape[-2], device=query.device).reshape(1, -1)
        allowed_mask = cols <= rows
    elif attn_mask is not None:
        _require_npu_tensor(attn_mask, "attn_mask")
        if attn_mask.dtype == torch.bool:
            allowed_mask = attn_mask
        else:
            scores = scores + attn_mask.to(dtype=scores.dtype)

    if allowed_mask is None:
        probabilities = torch.softmax(scores, dim=-1)
    else:
        mask_values = allowed_mask.to(dtype=scores.dtype)
        masked_scores = scores * mask_values + (1.0 - mask_values) * torch.finfo(scores.dtype).min
        row_max = masked_scores.max(dim=-1, keepdim=True).values
        exponents = torch.exp(masked_scores - row_max) * mask_values
        denominator = exponents.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(scores.dtype).tiny
        )
        probabilities = exponents / denominator
    return torch.matmul(probabilities, value)


__all__ = ["scaled_dot_product_attention_310b"]
