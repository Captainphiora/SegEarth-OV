import gc
import json

import torch
import torch.nn.functional as functional
import torch_npu

from npu_compat import setup_npu
from simfeatup_dev.upsamplers import _reflection_pad2d


setup_npu(use_patch=False)
torch.npu.set_compile_mode(jit_compile=False)
cpu_input = torch.randn(1, 4, 9, 11, dtype=torch.float16)
pad = (2, 1, 3, 2)
expected = functional.pad(cpu_input, pad, mode="reflect")
actual = _reflection_pad2d(cpu_input.to("npu:0"), pad).cpu()
error = float((actual - expected).abs().max().item())
if not torch.equal(actual, expected):
    raise AssertionError(error)
print(json.dumps({"call_site": "simfeatup._reflection_pad2d", "max_error": error}))
del cpu_input, expected, actual
gc.collect()
torch.npu.empty_cache()
torch.npu.synchronize()
