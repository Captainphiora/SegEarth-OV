"""Owned device-side Linspace for Ascend 310B."""

import functools

import torch
import torch_npu  # noqa: F401


@functools.lru_cache(maxsize=32)
def _signed_position_index(steps, dtype, device_name):
    device = torch.device(device_name)
    halfway = steps // 2
    left = torch.arange(halfway, dtype=dtype, device=device)
    right_count = steps - halfway
    right = torch.arange(right_count, dtype=dtype, device=device)
    right = right - (right_count - 1)
    return torch.cat((left, right))


def linspace_310b(start, end, steps, *, dtype=None, device="npu:0"):
    """Generate Linspace values on NPU without a CPU tensor or H2D value copy."""
    if not isinstance(steps, int):
        raise TypeError("steps must be an integer")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    target_device = torch.device(device)
    if target_device.type != "npu":
        raise ValueError("device must be an NPU device")
    output_dtype = torch.get_default_dtype() if dtype is None else dtype
    if output_dtype not in (torch.float16, torch.float32):
        raise TypeError("Ascend 310B Linspace supports float16 and float32 only")
    if steps == 0:
        return torch.empty((0,), dtype=output_dtype, device=target_device)
    if steps == 1 or float(start) == float(end):
        return torch.full((steps,), start, dtype=output_dtype, device=target_device)

    halfway = steps // 2
    values = _signed_position_index(
        steps, output_dtype, str(target_device)
    ).clone()
    values.mul_((float(end) - float(start)) / (steps - 1))
    values[:halfway].add_(float(start))
    values[halfway:].add_(float(end))
    return values


__all__ = ["linspace_310b"]
