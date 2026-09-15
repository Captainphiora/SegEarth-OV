"""Owned Ascend 310B ReflectionPad2d composition."""

import functools

import torch
import torch_npu  # noqa: F401


def _reflection_indices(length, before, after, device):
    left_steps = torch.arange(1, before + 1, dtype=torch.int64, device=device)
    left = before + 1 - left_steps
    center = torch.arange(length, dtype=torch.int64, device=device)
    right_steps = torch.arange(1, after + 1, dtype=torch.int64, device=device)
    right = length - 1 - right_steps
    return torch.cat((left, center, right))


@functools.lru_cache(maxsize=32)
def _flat_reflection_index(
    height, width, left, right, top, bottom, device_name
):
    device = torch.device(device_name)
    width_index = _reflection_indices(width, left, right, device)
    height_index = _reflection_indices(height, top, bottom, device)
    return (
        height_index.reshape(-1, 1) * width + width_index.reshape(1, -1)
    ).reshape(-1)


def reflection_pad2d_310b(input_tensor, pad):
    """Reflect-pad rank-4 NCHW input without ReflectionPad2d or Flip."""
    if not getattr(input_tensor, "is_npu", False):
        raise ValueError("input must be an NPU tensor")
    if input_tensor.dim() != 4:
        raise ValueError("input must be rank-4 NCHW")
    if input_tensor.dtype not in (torch.float16, torch.float32):
        raise TypeError("input dtype must be float16 or float32")
    if len(pad) != 4 or any(not isinstance(value, int) for value in pad):
        raise ValueError("pad must contain four integers")
    left, right, top, bottom = pad
    if min(pad) < 0:
        raise ValueError("negative reflection padding is unsupported")
    n, c, height, width = input_tensor.shape
    if left >= width or right >= width or top >= height or bottom >= height:
        raise ValueError("each pad value must be smaller than its input dimension")

    flat_index = _flat_reflection_index(
        height, width, left, right, top, bottom, str(input_tensor.device)
    )
    return input_tensor.reshape(n, c, height * width).index_select(
        2, flat_index
    ).reshape(n, c, height + top + bottom, width + left + right)


__all__ = ["reflection_pad2d_310b"]
