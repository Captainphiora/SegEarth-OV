"""Owned Ascend 310B Im2col/Unfold composition."""

import torch
import torch.nn.functional as functional
import torch_npu  # noqa: F401


def _pair(value, name, *, allow_zero=False):
    pair = (value, value) if isinstance(value, int) else tuple(value)
    if len(pair) != 2 or any(not isinstance(item, int) for item in pair):
        raise ValueError(f"{name} must be an integer or pair of integers")
    minimum = 0 if allow_zero else 1
    if any(item < minimum for item in pair):
        raise ValueError(f"{name} values must be >= {minimum}")
    return pair


def im2col_310b(input_tensor, kernel_size, dilation=1, padding=0, stride=1):
    """Return PyTorch Unfold ordering using NPU padding, indexing, and reshape."""
    if not getattr(input_tensor, "is_npu", False):
        raise ValueError("input must be an NPU tensor")
    if input_tensor.dim() != 4:
        raise ValueError("input must be rank-4 NCHW")
    if input_tensor.dtype not in (torch.float16, torch.float32):
        raise TypeError("input dtype must be float16 or float32")

    kh, kw = _pair(kernel_size, "kernel_size")
    dh, dw = _pair(dilation, "dilation")
    ph, pw = _pair(padding, "padding", allow_zero=True)
    sh, sw = _pair(stride, "stride")

    padded = input_tensor
    if ph or pw:
        padded = functional.pad(padded, (pw, pw, ph, ph), mode="constant", value=0)
    n, c, h, w = padded.shape
    oh = (h - dh * (kh - 1) - 1) // sh + 1
    ow = (w - dw * (kw - 1) - 1) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("kernel, dilation, and padding produce an empty output")

    stride_n, stride_c, stride_h, stride_w = padded.stride()
    windows = padded.as_strided(
        (n, c, oh, ow, kh, kw),
        (
            stride_n,
            stride_c,
            sh * stride_h,
            sw * stride_w,
            dh * stride_h,
            dw * stride_w,
        ),
    )
    output = windows.permute(0, 1, 4, 5, 2, 3).reshape(
        n, c * kh * kw, oh * ow
    )
    return output.clone() if output.data_ptr() == input_tensor.data_ptr() else output


__all__ = ["im2col_310b"]
