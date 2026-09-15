# Im2col / Unfold Status

- contract: `../contracts/im2col.md`
- phase: performance optimization and final operator/integration validation passed
- source state: FP16/FP32 `as_strided` window view plus one materializing reshape;
  the identity-view case clones to preserve fresh-output semantics
- latest command: optimized operator and `adaptive_conv_py_simple` integration passed
- latest evidence: paired A/B baseline 16047.7 us, candidate 5275.9 us, 3.04x;
  three operator cases exact, integration max error 0.00390625
- cleanup evidence: probe, candidate, and integration guards passed; Swap 0
- next action: preserve current source; full demo output passed, pending device reboot cleanup
