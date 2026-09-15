# Linspace Status

- contract: `../contracts/linspace.md`
- phase: expanded correctness and final operator/integration validation passed
- source state: bounded signed-position metadata cache with fresh per-call output;
  FP16/FP32 endpoint arithmetic remains entirely on NPU
- latest command: six-case final operator test and `SimpleImplicitFeaturizer.forward`
- latest evidence: new 258-point FP16 tail case reduced from a tolerance failure to
  max error 0; integration max error 0.0009689331. Paired performance is 0.70x
  (3979.2 us baseline, 5705.7 us exact candidate), accepted only for correctness
- cleanup evidence: all Linspace probes, final test, and integration guards passed; Swap 0
- next action: preserve correctness path; fused owned kernel is needed for speed recovery
