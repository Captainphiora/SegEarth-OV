# ReflectionPad2d NPU Validation - 2026-08-21

The native `F.pad(..., mode="reflect")` probe failed in
`aclnnReflectionPad2d`/MirrorPad with error `561103` because its dynamic kernel
configuration could not be loaded. Cleanup passed (`3328 -> 3425 MiB`, Swap 0).

The first candidate used NPU `torch.where` to form indices and failed because
SelectV2 has no 310B binary. That assumption was removed rather than hidden by a
fallback. Cleanup passed (`3411 -> 3528 MiB`, Swap 0).

The accepted candidate forms left/center/right indices with positive NPU Range,
integer arithmetic, and Cat, then uses IndexSelect. It passed three FP16 cases
with maximum error 0. Timed on `[1,8,31,29]` with pad `(3,2,4,1)`, 3 warmups and
10 synchronized samples gave median 5126.1715 us and p95 7711.627 us. Cleanup
passed (`3414 -> 3533 MiB`, Swap 0).

The reflect-pad paths in `simfeatup_dev/upsamplers.py` now dispatch NPU tensors
to this candidate while CPU/GPU tensors retain `F.pad`. The integration probe
matched the CPU reference exactly. Guard cleanup passed (`3534 -> 3650 MiB`,
Swap 0).
