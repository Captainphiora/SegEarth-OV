# Im2col NPU Validation - 2026-08-21

The native `F.unfold` probe failed in `aclnnIm2col` with error `561103` because
the Im2col dynamic AICore kernel could not be added to the launcher. Cleanup
passed with NPU memory `3417 -> 3529 MiB` and Swap 0.

The candidate uses only constant NPU padding, device Range/index arithmetic,
IndexSelect, reshape, and permute. It passed three FP16 cases spanning scalar and
pair kernel/dilation/padding/stride attributes with exact output ordering and
maximum error 0. On input `[1,8,32,32]`, kernel 3, padding 1, stride 1, 3 warmups
and 10 synchronized samples gave median 9074.673 us and p95 10965.867 us.
Cleanup passed (`3430 -> 3551 MiB`, Swap 0).

Source audit found existing `torch.nn.Unfold` use in
`simfeatup_dev/upsamplers.py`; its NPU path now dispatches to this candidate.
The real `adaptive_conv_py_simple` integration probe passed against the CPU
reference with maximum error 0.00390625. Guard cleanup passed
(`3551 -> 3643 MiB`, Swap 0).
