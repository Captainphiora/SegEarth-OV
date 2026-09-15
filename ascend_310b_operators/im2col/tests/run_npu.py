import gc
import json
import statistics
import time

import torch
import torch.nn.functional as functional
import torch_npu

from ascend_310b_operators.im2col.source import im2col_310b
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
    ((1, 1, 5, 7), 1, 1, 0, 1),
    ((2, 3, 9, 11), (3, 2), 1, (1, 2), (2, 1)),
    ((1, 2, 13, 10), (2, 3), (2, 1), 0, (3, 2)),
]
max_error = 0.0
for shape, kernel, dilation, padding, stride in cases:
    cpu_input = torch.randn(shape, dtype=torch.float16)
    actual = im2col_310b(cpu_input.to("npu:0"), kernel, dilation, padding, stride).cpu()
    expected = functional.unfold(cpu_input, kernel, dilation, padding, stride)
    error = float((actual - expected).abs().max().item())
    max_error = max(max_error, error)
    if not torch.equal(actual, expected):
        raise AssertionError((shape, kernel, dilation, padding, stride, error))

bench_input = torch.randn(1, 8, 32, 32, dtype=torch.float16, device="npu:0")
timing = benchmark(lambda: im2col_310b(bench_input, 3, padding=1, stride=1))
print(json.dumps({"operator": "im2col", "cases": len(cases), "max_error": max_error, **timing}))
del bench_input
gc.collect()
torch.npu.empty_cache()
torch.npu.synchronize()
