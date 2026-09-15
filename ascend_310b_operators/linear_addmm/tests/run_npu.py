import gc
import json
import statistics
import time

import torch
import torch.nn.functional as functional
import torch_npu

from ascend_310b_operators.linear_addmm.source import linear_310b
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
torch.manual_seed(7)
cases = [((3, 16), (12, 16), True), ((2, 5, 17), (13, 17), False), ((1, 7, 31), (19, 31), True)]
max_error = 0.0
for input_shape, weight_shape, with_bias in cases:
    cpu_input = torch.randn(input_shape, dtype=torch.float16)
    cpu_weight = torch.randn(weight_shape, dtype=torch.float16)
    cpu_bias = torch.randn(weight_shape[0], dtype=torch.float16) if with_bias else None
    npu_bias = None if cpu_bias is None else cpu_bias.to("npu:0")
    actual = linear_310b(cpu_input.to("npu:0"), cpu_weight.to("npu:0"), npu_bias).cpu()
    expected = functional.linear(cpu_input, cpu_weight, cpu_bias)
    error = float((actual - expected).abs().max().item())
    max_error = max(max_error, error)
    if not torch.allclose(actual, expected, rtol=1e-2, atol=1e-2):
        raise AssertionError((input_shape, weight_shape, with_bias, error))

bench_input = torch.randn(64, 256, dtype=torch.float16, device="npu:0")
bench_weight = torch.randn(256, 256, dtype=torch.float16, device="npu:0")
bench_bias = torch.randn(256, dtype=torch.float16, device="npu:0")
timing = benchmark(lambda: linear_310b(bench_input, bench_weight, bench_bias))
print(json.dumps({"operator": "linear", "cases": len(cases), "max_error": max_error, **timing}))
del bench_input, bench_weight, bench_bias
gc.collect()
torch.npu.empty_cache()
torch.npu.synchronize()
