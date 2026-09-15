# Linear NPU Validation - 2026-08-21

- Device/environment: Ascend 310B1 physical 0, SegEarth, CANN 9.0.0
- Candidate: `source.linear_310b`, FP16 matmul followed by optional bias add
- Patches: disabled with `setup_npu(use_patch=False)`

The native `F.linear` probe failed in `aclnnAddmm` with error `561103`: no
`MatMulV2` binary exists for the observed FP16 `FRACTAL_NZ` integral key. Its
guard cleanup passed with NPU memory `3307 -> 3408 MiB` and Swap 0.

The candidate passed three 2-D/batched/tail cases with and without bias. Maximum
absolute error was 0.0078125. For FP16 input `[64,256]`, weight `[256,256]`, and
bias `[256]`, 3 warmups plus 10 synchronized samples produced median 524.1055 us
and p95 14279.877 us. Candidate cleanup passed with NPU memory
`3288 -> 3392 MiB` and Swap 0. The high p95 remains a performance risk.

Only Linear semantics are claimed; full Addmm alpha/beta semantics are not.
The NPU branch of `open_clip.modified_resnet.AttentionPool2d` now uses
`linear_310b` for Q/K/V/output projection. The combined Linear plus SDPA
integration probe passed against the CPU reference with maximum error
0.000244140625. Guard cleanup passed (`3548 -> 3649 MiB`, Swap 0).
