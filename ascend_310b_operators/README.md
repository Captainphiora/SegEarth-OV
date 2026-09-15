# Ascend 310B Operators

This directory turns the six entries in `../PATCH_LIST.md` into explicit
operator work items for the Atlas 200I DK A2 (Ascend 310B1). The existing Python
patches keep SegEarth running; they do not count as finished operators.

| Operator workspace | Target API | Current compatibility path |
|---|---|---|
| `scaled_dot_product_attention/` | `F.scaled_dot_product_attention` | math decomposition |
| `linear_addmm/` | `F.linear` / Addmm semantics | `matmul` plus `add` |
| `linspace/` | `torch.linspace` | CPU generation then copy |
| `reflection_pad2d/` | `F.pad(..., mode="reflect")` | NPU `index_select` composition |
| `im2col/` | `F.unfold` | NPU indexing and reshape composition |
| `upsample_bilinear2d/` | `F.interpolate(..., mode="bilinear")` | separable NPU composition |

Each operator's required semantics and minimum validation matrix live under
`contracts/`. Create its source and evidence directories only when work begins.
Repository-wide execution and no-fallback rules are in `AGENTS.md`.

The discoverable Codex skill is
`../.codex/skills/ascend-310b-operators/SKILL.md`. Use that skill with exactly one
selected operator at a time.

Environment snapshot verified on 2026-08-21: CANN 9.0.0, Bisheng based on Clang
15.0.5, `torch_npu` 2.8.0.post4, and Ascend 310B1 physical device 0. Re-check
these values for every evidence run; the snapshot is not a permanent default.
