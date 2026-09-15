# SDPA NPU Validation - 2026-08-21

The native `F.scaled_dot_product_attention` probe entered
`aclnnFlashAttentionScore` and failed with error `161001`; FlashAttentionScore
has no usable InferShape implementation on Ascend 310B. Cleanup passed
(`3435 -> 3560 MiB`, Swap 0).

The accepted FP16 inference candidate owns the matmul, scaling, mask, softmax,
and output matmul dataflow. Boolean/causal masking uses masked exponentiation and
denominator clamp, avoiding unsupported SelectV2 and returning zero for fully
masked rows. Dropout other than zero fails closed.

Four cases covering self/cross attention, causal masking, GQA, a boolean mask,
and a fully masked row passed with maximum error 0.0009765625. On Q/K/V shape
`[1,4,64,64]`, 3 warmups and 10 synchronized samples gave median 496.8975 us
and p95 971.732 us. Cleanup passed (`3484 -> 3591 MiB`, Swap 0).

The NPU branch of `open_clip.modified_resnet.AttentionPool2d` now explicitly
uses this candidate for attention and `linear_310b` for projections. A CPU
reference versus FP16 NPU integration probe passed with maximum error
0.000244140625 while compatibility patches were disabled. Its guard cleanup
passed (`3548 -> 3649 MiB`, Swap 0).
