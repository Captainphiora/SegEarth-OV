import gc
import json
import statistics
import time

import torch
import torch.nn.functional as functional
import torch_npu

from ascend_310b_operators.reflection_pad2d.source import reflection_pad2d_310b
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
cases = [((1, 1, 2, 2), (0, 0, 0, 0)), ((1, 2, 3, 5), (1, 2, 1, 1)), ((2, 3, 7, 9), (2, 1, 3, 2))]
max_error = 0.0
for shape, pad in cases:
    cpu_input = torch.arange(torch.tensor(shape).prod()).reshape(shape).to(torch.float16)
    actual = reflection_pad2d_310b(cpu_input.to("npu:0"), pad).cpu()
    expected = functional.pad(cpu_input, pad, mode="reflect")
    error = float((actual - expected).abs().max().item())
    max_error = max(max_error, error)
    if not torch.equal(actual, expected):
        raise AssertionError((shape, pad, error))

bench_input = torch.randn(1, 8, 31, 29, dtype=torch.float16, device="npu:0")
timing = benchmark(lambda: reflection_pad2d_310b(bench_input, (3, 2, 4, 1)))
print(json.dumps({"operator": "reflection_pad2d", "cases": len(cases), "max_error": max_error, **timing}))
del bench_input
gc.collect()
torch.npu.empty_cache()
torch.npu.synchronize()
