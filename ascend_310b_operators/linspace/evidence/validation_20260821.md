# Linspace NPU Validation - 2026-08-21

- Device: Ascend 310B1, physical 0, logical `npu:0`
- Environment: SegEarth, CANN 9.0.0, torch_npu 2.8.0.post4
- Candidate: `source.linspace_310b`, device-side `arange` and arithmetic
- Patches during validation: disabled with `setup_npu(use_patch=False)`

## Native API Probe

`torch.linspace(-1, 1, 7, dtype=float16, device="npu:0")` failed through
`aclnnLinspace` with error `161002` and
`support for Ascend310B is not implemented`. Guard cleanup passed with NPU memory
`3284 -> 3382 MiB` and Swap 0.

## Candidate Result

- Cases: 5, including steps 0/1, reversed endpoints, FP16 and FP32
- Maximum absolute error: 0.0009765625
- Timed case: 257 FP16 elements, 3 warmups and 10 synchronized samples
- Median: 628.596 us
- P95: 2437.445 us
- Guard cleanup: passed, NPU memory `3269 -> 3397 MiB`, Swap 0

The candidate path creates no CPU tensor and performs no host-to-device value
copy. `simfeatup_dev.upsamplers.SimpleImplicitFeaturizer.forward` now dispatches
its NPU coordinate generation to this candidate. The CPU-reference integration
probe passed with maximum error 0.002562582492828369. Guard cleanup passed
(`3532 -> 3655 MiB`, Swap 0).
