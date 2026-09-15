"""Owned Ascend 310B Linear composition."""

import torch
import torch_npu  # noqa: F401


def _require_npu_tensor(tensor, name):
    if not getattr(tensor, "is_npu", False):
        raise ValueError(f"{name} must be an NPU tensor")


def linear_310b(input_tensor, weight, bias=None):
    """Compute ``input @ weight.T + bias`` without calling Linear or Addmm."""
    _require_npu_tensor(input_tensor, "input")
    _require_npu_tensor(weight, "weight")
    if input_tensor.dtype != torch.float16 or weight.dtype != input_tensor.dtype:
        raise TypeError("Ascend 310B Linear currently supports matching float16 inputs only")
    if input_tensor.dim() < 1 or weight.dim() != 2:
        raise ValueError("input must have rank >= 1 and weight must have rank 2")
    if input_tensor.shape[-1] != weight.shape[-1]:
        raise ValueError("input and weight K dimensions must match")
    if bias is not None:
        _require_npu_tensor(bias, "bias")
        if bias.dtype != input_tensor.dtype or bias.dim() != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError("bias must be float16 with shape [out_features]")

    output = torch.matmul(input_tensor, weight.transpose(-2, -1))
    return output if bias is None else output + bias


__all__ = ["linear_310b"]
