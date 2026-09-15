# Linspace Contract

## Identity

- Target API: `torch.linspace` on `npu:0`.
- Missing path: `aclnnLinspace` is unavailable in the observed 310B runtime.
- Invalid final workaround: `npu_compat.py:104-117` generates on CPU and copies.

## Required Semantics

Generate all elements on the NPU, including endpoint behavior. For `steps > 1`,
element `i` is `start + i * (end - start) / (steps - 1)` with the declared dtype
rounding. `steps == 0` returns empty and `steps == 1` returns `start`. Reject
unsupported scalar/dtype combinations explicitly.

The candidate path must not allocate a CPU tensor, call CPU `torch.linspace`, or
perform a host-to-device copy of generated values.

## Minimum Validation Matrix

- FP16 and FP32 if claimed; positive, negative, reversed, and equal endpoints.
- `steps` 0, 1, 2, odd, even, and non-vector-aligned tail lengths.
- Integer endpoints with floating output and explicit dtype selection.
- Endpoint, monotonicity, maximum error, output dtype, shape, and device checks.
- Real call sites from `simfeatup_dev/upsamplers.py` and positional-embedding code.
