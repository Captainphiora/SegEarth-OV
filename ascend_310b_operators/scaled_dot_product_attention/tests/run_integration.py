import copy
import gc
import json

import torch
import torch_npu

from npu_compat import setup_npu
from open_clip.modified_resnet import AttentionPool2d


setup_npu(use_patch=False)
torch.npu.set_compile_mode(jit_compile=False)
torch.manual_seed(19)
cpu_module = AttentionPool2d(spacial_dim=3, embed_dim=32, num_heads=4, output_dim=24).eval()
npu_module = copy.deepcopy(cpu_module).to(dtype=torch.float16, device="npu:0")
cpu_input = torch.randn(2, 32, 3, 3)
expected = cpu_module(cpu_input)
actual = npu_module(cpu_input.to(dtype=torch.float16, device="npu:0")).cpu()
error = float((actual - expected.to(torch.float16)).abs().max().item())
if not torch.allclose(actual, expected.to(torch.float16), rtol=3e-2, atol=3e-2):
    raise AssertionError(error)
print(json.dumps({"call_site": "open_clip.AttentionPool2d", "max_error": error}))
del cpu_module, npu_module, cpu_input, expected, actual
gc.collect()
torch.npu.empty_cache()
torch.npu.synchronize()
