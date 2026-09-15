# UpsampleBilinear2d NPU Validation - 2026-08-21

The native boundary input `[2,3,5,7]`, output `(1,4)`, and
`align_corners=True` failed in `aclnnUpsampleBilinear2d` with error `561103`
because `ResizeBilinearV2AiCore` could not be loaded. Cleanup passed
(`3429 -> 3526 MiB`, Swap 0).

The candidate computes coordinates and weights on NPU and performs separable
height/width IndexSelect interpolation. It passed four FP16 cases covering
explicit size, scale factor, both align-corners modes, output dimension one, and
identity size. Maximum error was 0.00146484375. On `[1,8,32,32]` to
`[1,8,64,64]`, 3 warmups and 10 synchronized samples gave median 11903.13 us
and p95 20226.02 us. Cleanup passed (`3431 -> 3544 MiB`, Swap 0).

`antialias=True` fails closed and bicubic is not routed to bilinear. The NPU
branch of `simfeatup_dev.upsamplers.Bilinear.forward` now calls this candidate;
its CPU-reference integration probe passed with maximum error 0.001953125.
Guard cleanup passed (`3529 -> 3647 MiB`, Swap 0).
