import gc
import json
import statistics
import time

import torch
import torch.nn.functional as functional
import torch_npu

from ascend_310b_operators.upsample_bilinear2d.source import upsample_bilinear2d_310b
from npu_compat import setup_npu


def benchmark(callable_):
    for _ in range(3):
        callable_()
    torch.npu.synchronize()
    samples = []
    for _ in range(10):
        start = time.perf_counter_ns()
        callable_()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1000.0)
    ordered = sorted(samples)
    return {"median_us": statistics.median(samples), "p95_us": ordered[8]}


setup_npu(use_patch=False)
torch.npu.set_compile_mode(jit_compile=False)
cases = [
    ((1, 1, 3, 5), {"size": (7, 9), "align_corners": False}),
    ((2, 3, 5, 7), {"size": (1, 4), "align_corners": True}),
    ((1, 2, 9, 11), {"scale_factor": (1.5, 2.0), "align_corners": False}),
    ((1, 1, 8, 6), {"size": (8, 6), "align_corners": False}),
]
max_error = 0.0
for shape, kwargs in cases:
    cpu_input = torch.randn(shape, dtype=torch.float16)
    actual = upsample_bilinear2d_310b(cpu_input.to("npu:0"), **kwargs).cpu()
    expected = functional.interpolate(cpu_input, mode="bilinear", **kwargs)
    error = float((actual - expected).abs().max().item())
    max_error = max(max_error, error)
    if not torch.allclose(actual, expected, rtol=2e-3, atol=2e-3):
        raise AssertionError((shape, kwargs, error))

bench_input = torch.randn(1, 8, 32, 32, dtype=torch.float16, device="npu:0")
timing = benchmark(lambda: upsample_bilinear2d_310b(bench_input, size=(64, 64)))
print(json.dumps({"operator": "upsample_bilinear2d", "cases": len(cases), "max_error": max_error, **timing}))
del bench_input
gc.collect()
torch.npu.empty_cache()
torch.npu.synchronize()
