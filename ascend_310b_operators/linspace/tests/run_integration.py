import gc
import json

import torch
import torch_npu

from npu_compat import setup_npu
from simfeatup_dev.upsamplers import SimpleImplicitFeaturizer


setup_npu(use_patch=False)
torch.npu.set_compile_mode(jit_compile=False)
module = SimpleImplicitFeaturizer(n_freqs=4).eval()
cpu_input = torch.randn(1, 3, 5, 7)
expected = module(cpu_input)
actual = module(cpu_input.to(dtype=torch.float16, device="npu:0")).cpu()
expected = expected.to(actual.dtype)
error = float((actual - expected).abs().max().item())
if not torch.allclose(actual, expected, rtol=3e-2, atol=3e-2):
    raise AssertionError(error)
print(json.dumps({"call_site": "SimpleImplicitFeaturizer.forward", "max_error": error}))
del module, cpu_input, expected, actual
gc.collect()
torch.npu.empty_cache()
torch.npu.synchronize()
