# UpsampleBilinear2d Contract

## Identity

- Target API: `F.interpolate(..., mode="bilinear")` for rank-4 NCHW tensors.
- Missing path: `aclnnUpsampleBilinear2d` fails for part of the observed 310B
  shape/attribute domain.
- Compatibility evidence: `npu_compat.py:185-247` uses separable NPU indexing and
  interpolation after trying the original operator.

## Required Semantics

Implement bilinear coordinate mapping and interpolation for both
`align_corners=False` and `True`, preserving NCHW shape and dtype. Support either
explicit `size` or the documented `scale_factor` domain, including correct output
size rounding. State the `recompute_scale_factor` and `antialias` support boundary.

Bicubic is a different operator and must never be silently routed to bilinear.
Do not use `try/except` to call the original Interpolate first, and do not call a
vendor whole UpsampleBilinear2d or CPU resize as the candidate implementation.

## Minimum Validation Matrix

- FP16 and any additionally claimed dtype; batch/channel greater than one.
- Upsampling, downsampling if claimed, identity size, non-square input/output, and
  output dimensions 1 and greater than 1.
- Explicit size and scalar/pair scale factors; `align_corners` false and true.
- Border/corner values, half-pixel locations, non-aligned tails, output dtype and
  shape, with tolerance against an untimed CPU golden.
- Real bilinear call sites in `segearth_segmentor.py`, `gem/`, and
  `simfeatup_dev/upsamplers.py` with the patch disabled.
