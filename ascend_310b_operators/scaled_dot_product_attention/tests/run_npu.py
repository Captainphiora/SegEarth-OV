import gc
import json
import statistics
import time

import torch
import torch.nn.functional as functional
import torch_npu

from ascend_310b_operators.scaled_dot_product_attention.source import scaled_dot_product_attention_310b
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
torch.manual_seed(11)
case_specs = [
    ((1, 2, 7, 16), (1, 2, 7, 16), {}),
    ((2, 3, 5, 16), (2, 3, 9, 16), {"is_causal": True}),
    ((1, 4, 6, 16), (1, 2, 8, 16), {"enable_gqa": True, "scale": 0.2}),
]
max_error = 0.0
for query_shape, key_shape, kwargs in case_specs:
    value_shape = key_shape[:-1] + (12,)
    query = torch.randn(query_shape, dtype=torch.float16)
    key = torch.randn(key_shape, dtype=torch.float16)
    value = torch.randn(value_shape, dtype=torch.float16)
    actual = scaled_dot_product_attention_310b(
        query.to("npu:0"), key.to("npu:0"), value.to("npu:0"), **kwargs
    ).cpu()
    expected = functional.scaled_dot_product_attention(
        query.to(torch.float32), key.to(torch.float32), value.to(torch.float32), **kwargs
    ).to(torch.float16)
    error = float((actual - expected).abs().max().item())
    max_error = max(max_error, error)
    if not torch.allclose(actual, expected, rtol=3e-2, atol=3e-2):
        raise AssertionError((query_shape, key_shape, kwargs, error))

query = torch.randn(1, 2, 5, 16, dtype=torch.float16)
key = torch.randn(1, 2, 5, 16, dtype=torch.float16)
value = torch.randn(1, 2, 5, 8, dtype=torch.float16)
mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)
mask[..., -1, :] = False
actual = scaled_dot_product_attention_310b(
    query.to("npu:0"), key.to("npu:0"), value.to("npu:0"), attn_mask=mask.to("npu:0")
).cpu()
expected = functional.scaled_dot_product_attention(
    query.to(torch.float32), key.to(torch.float32), value.to(torch.float32), attn_mask=mask
).to(torch.float16)
mask_error = float((actual - expected).abs().max().item())
max_error = max(max_error, mask_error)
if not torch.allclose(actual, expected, rtol=3e-2, atol=3e-2):
    raise AssertionError(("boolean_mask", mask_error))

bench_query = torch.randn(1, 4, 64, 64, dtype=torch.float16, device="npu:0")
bench_key = torch.randn(1, 4, 64, 64, dtype=torch.float16, device="npu:0")
bench_value = torch.randn(1, 4, 64, 64, dtype=torch.float16, device="npu:0")
timing = benchmark(
    lambda: scaled_dot_product_attention_310b(bench_query, bench_key, bench_value)
)
print(json.dumps({"operator": "sdpa", "cases": len(case_specs) + 1, "max_error": max_error, **timing}))
del bench_query, bench_key, bench_value
gc.collect()
torch.npu.empty_cache()
torch.npu.synchronize()
