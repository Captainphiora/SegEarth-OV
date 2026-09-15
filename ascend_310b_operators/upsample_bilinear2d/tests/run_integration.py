import gc
import json

import torch
import torch_npu

from npu_compat import setup_npu
from simfeatup_dev.upsamplers import Bilinear


setup_npu(use_patch=False)
torch.npu.set_compile_mode(jit_compile=False)
module = Bilinear().eval()
cpu_features = torch.randn(1, 4, 5, 7)
cpu_image = torch.randn(1, 3, 11, 13)
expected = module(cpu_features, cpu_image)
actual = module(
    cpu_features.to(dtype=torch.float16, device="npu:0"),
    cpu_image.to(dtype=torch.float16, device="npu:0"),
).cpu()
error = float((actual - expected.to(torch.float16)).abs().max().item())
if not torch.allclose(actual, expected.to(torch.float16), rtol=3e-2, atol=3e-2):
    raise AssertionError(error)
print(json.dumps({"call_site": "simfeatup.Bilinear.forward", "max_error": error}))
del module, cpu_features, cpu_image, expected, actual
gc.collect()
torch.npu.empty_cache()
torch.npu.synchronize()
