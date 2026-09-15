# Scaled Dot Product Attention Status

- contract: `../contracts/scaled_dot_product_attention.md`
- phase: correctness and final operator/full-demo integration passed; speed candidate rejected
- source state: FP16 composition revised to avoid unsupported SelectV2/`torch.where`
- latest command: post-reboot flatten-bmm paired A/B probe
- latest evidence: four correctness cases passed at max error 0.0009765625;
  paired trials were 0.75x and 3.27x, so the 1.002x combined result was rejected
- cleanup evidence: final independent operator and integration guards passed with
  Swap 0; the full-demo command completed but requires device reboot for cleanup
- next action: preserve current source; an owned fused attention kernel is required
- warmed level-1 evidence: one full warmup then five identical-output predictions;
  median 25.258889 s, max 25.265202 s, cgroup peak 4308 MiB
- warmed cleanup: NPU 6719 -> 6666 MiB, but global Swap increased to 256 KiB;
  strict cleanup gate failed despite no surviving process/cgroup
- patch-on warmed comparison: all six global patches enabled; five-run median
  25.543709 s versus patch-off 25.258889 s, so patch-off is 1.13% faster;
  identical output tensor SHA256 and patch-on cgroup peak 5735 MiB
- patch-on cleanup: Swap remained 0, but NPU 2261 -> 2812 MiB exceeded the
  128 MiB normal tolerance; audited `cleanup_npu.sh` fallback completed with no
  process targets, final NPU 2102 MiB, Swap 0, and no residual cgroup
