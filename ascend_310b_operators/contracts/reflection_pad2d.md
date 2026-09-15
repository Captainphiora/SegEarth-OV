# ReflectionPad2d Contract

## Identity

- Target API: `F.pad(input, pad, mode="reflect")` for rank-4 image tensors.
- Missing path: observed `aclnnReflectionPad2d`/`aclnnFlip` support is unusable.
- Compatibility evidence: `npu_compat.py:120-142` builds indices and uses two
  NPU `index_select` operations.

## Required Semantics

For pad `(left, right, top, bottom)`, mirror values without repeating the edge.
Each horizontal pad must be smaller than input width and each vertical pad
smaller than input height. Preserve NCHW batch/channel dimensions, dtype, layout
contract, and PyTorch error behavior for invalid pads.

Do not call `F.pad(..., reflect)`, ReflectionPad2d, Flip, or a CPU indexing path as
the whole candidate implementation.

## Minimum Validation Matrix

- FP16 and any additionally claimed dtype on rank-4 NCHW inputs.
- Symmetric, asymmetric, one-sided, zero, and maximum-valid padding.
- Height/width 2, non-square images, odd sizes, and vector-tail dimensions.
- Invalid pad equal to or greater than its input dimension must fail clearly.
- Exact value comparison where dtype permits, including corners and edges.
- The reflect-pad call sites in `simfeatup_dev/upsamplers.py` with patch disabled.
