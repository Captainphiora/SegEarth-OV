import gc
import json
import statistics
import time

import torch
import torch_npu

from ascend_310b_operators.linspace.source import linspace_310b
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
    (-1.0, 1.0, 0, torch.float16),
    (-1.0, 1.0, 1, torch.float16),
    (2.0, -3.0, 7, torch.float16),
    (0.25, 0.25, 17, torch.float32),
    (-2.0, 10.0, 33, torch.float32),
    (-7.5, 3.25, 258, torch.float16),
]
max_error = 0.0
for start, end, steps, dtype in cases:
    actual = linspace_310b(start, end, steps, dtype=dtype).cpu()
    expected = torch.linspace(start, end, steps, dtype=dtype)
    error = 0.0 if steps == 0 else float((actual - expected).abs().max().item())
    max_error = max(max_error, error)
    if not torch.allclose(actual, expected, rtol=1e-3, atol=1e-3):
        raise AssertionError((start, end, steps, dtype, error))

timing = benchmark(lambda: linspace_310b(-1.0, 1.0, 257, dtype=torch.float16))
print(json.dumps({"operator": "linspace", "cases": len(cases), "max_error": max_error, **timing}))
gc.collect()
torch.npu.empty_cache()
torch.npu.synchronize()
