import gc
import json

import torch
import torch_npu

from npu_compat import setup_npu
from simfeatup_dev.upsamplers import adaptive_conv_py_simple


setup_npu(use_patch=False)
torch.npu.set_compile_mode(jit_compile=False)
torch.manual_seed(23)
cpu_input = torch.randn(1, 2, 5, 5)
cpu_filters = torch.randn(1, 3, 3, 3, 3)
expected = adaptive_conv_py_simple(cpu_input.clone(), cpu_filters.clone())
actual = adaptive_conv_py_simple(
    cpu_input.to(dtype=torch.float16, device="npu:0"),
    cpu_filters.to(dtype=torch.float16, device="npu:0"),
).cpu()
error = float((actual - expected.to(torch.float16)).abs().max().item())
if not torch.allclose(actual, expected.to(torch.float16), rtol=3e-2, atol=3e-2):
    raise AssertionError(error)
print(json.dumps({"call_site": "adaptive_conv_py_simple", "max_error": error}))
del cpu_input, cpu_filters, expected, actual
gc.collect()
torch.npu.empty_cache()
torch.npu.synchronize()
