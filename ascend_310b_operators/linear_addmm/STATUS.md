# Linear / Addmm Status

- contract: `../contracts/linear_addmm.md`
- phase: correctness and final operator/full-demo integration passed; speed candidates rejected
- source state: owned FP16 Linear composition in `source/__init__.py`; Addmm not claimed
- latest command: same-process `matmul` plus in-place bias candidate paired A/B
- latest evidence: flatten-mm candidate 0.28x; conservative in-place-bias candidate
  0.69x; both rejected and source unchanged
- cleanup evidence: final independent operator guard passed with Swap 0; the
  full-demo command completed but requires device reboot for system NPU cleanup
- next action: preserve current source; fused owned kernel is required for speedup
