"""Owned Ascend 310B bilinear resize composition."""

import functools
import math

import torch
import torch_npu  # noqa: F401


def _size_pair(value, name):
    pair = (value, value) if isinstance(value, (int, float)) else tuple(value)
    if len(pair) != 2:
        raise ValueError(f"{name} must be a scalar or pair")
    return pair


def _coordinates(input_size, output_size, align_corners, inverse_scale, device):
    output = torch.arange(output_size, dtype=torch.float32, device=device)
    if align_corners:
        scale = (input_size - 1) / (output_size - 1) if output_size > 1 else 0.0
        return output * scale
    return ((output + 0.5) * inverse_scale - 0.5).clamp(0.0, float(input_size - 1))


@functools.lru_cache(maxsize=32)
def _interpolation_metadata(
    input_h,
    input_w,
    output_h,
    output_w,
    align_corners,
    inverse_h,
    inverse_w,
    dtype,
    device_name,
):
    device = torch.device(device_name)
    source_h = _coordinates(input_h, output_h, align_corners, inverse_h, device)
    source_w = _coordinates(input_w, output_w, align_corners, inverse_w, device)
    h0 = source_h.floor().to(torch.int64)
    h1 = (h0 + 1).clamp(max=input_h - 1)
    w0 = source_w.floor().to(torch.int64)
    w1 = (w0 + 1).clamp(max=input_w - 1)
    weight_h = (source_h - h0.to(torch.float32)).to(dtype).reshape(
        1, 1, output_h, 1
    )
    weight_w = (source_w - w0.to(torch.float32)).to(dtype).reshape(
        1, 1, 1, output_w
    )
    return h0, h1, w0, w1, weight_h, weight_w


def upsample_bilinear2d_310b(
    input_tensor,
    size=None,
    scale_factor=None,
    *,
    align_corners=False,
    recompute_scale_factor=None,
    antialias=False,
):
    """Resize rank-4 NCHW input with explicit NPU bilinear interpolation."""
    if not getattr(input_tensor, "is_npu", False):
        raise ValueError("input must be an NPU tensor")
    if input_tensor.dim() != 4:
        raise ValueError("input must be rank-4 NCHW")
    if input_tensor.dtype not in (torch.float16, torch.float32):
        raise TypeError("input dtype must be float16 or float32")
    if (size is None) == (scale_factor is None):
        raise ValueError("provide exactly one of size or scale_factor")
    if antialias:
        raise ValueError("antialias=True is not supported by this 310B operator")

    input_h, input_w = input_tensor.shape[-2:]
    if size is not None:
        output_h, output_w = (int(value) for value in _size_pair(size, "size"))
        inverse_h = input_h / output_h
        inverse_w = input_w / output_w
    else:
        scale_h, scale_w = (float(value) for value in _size_pair(scale_factor, "scale_factor"))
        if scale_h <= 0 or scale_w <= 0:
            raise ValueError("scale_factor values must be positive")
        output_h = math.floor(input_h * scale_h)
        output_w = math.floor(input_w * scale_w)
        if recompute_scale_factor:
            inverse_h = input_h / output_h
            inverse_w = input_w / output_w
        else:
            inverse_h = 1.0 / scale_h
            inverse_w = 1.0 / scale_w
    if output_h <= 0 or output_w <= 0:
        raise ValueError("output spatial dimensions must be positive")

    h0, h1, w0, w1, weight_h, weight_w = _interpolation_metadata(
        input_h,
        input_w,
        output_h,
        output_w,
        bool(align_corners),
        float(inverse_h),
        float(inverse_w),
        input_tensor.dtype,
        str(input_tensor.device),
    )

    low_h = input_tensor.index_select(2, h0)
    high_h = input_tensor.index_select(2, h1)
    interpolated_h = low_h + (high_h - low_h) * weight_h
    low_w = interpolated_h.index_select(3, w0)
    high_w = interpolated_h.index_select(3, w1)
    return low_w + (high_w - low_w) * weight_w


__all__ = ["upsample_bilinear2d_310b"]
