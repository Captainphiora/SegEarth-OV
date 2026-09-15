# SegEarth Demo num_levels Validation - 2026-08-21

## Fixed Inputs

- Device: Atlas 200I DK A2, Ascend 310B1, physical 0 / logical `npu:0`
- Environment: `SegEarth`, CANN 9.0.0, torch_npu 2.8.0.post4
- Demo SHA256: `1956a712b3a024c79db4f5f679758375afc4e1e1a925577d46ee522cea7b233e`
- Input SHA256: `070a1ca21a3ae5fce378b63ce4d81eb7d12f8acddf401bbf5c98f226d0e26913`
- JBU weight SHA256: `cabc594d0042535f3413ac89d5f0b8b3173aecf18e2e469fb91b015ea4de49d8`
- Golden SHA256: `6c88526ba0e3c2b05845d66c7d81e756ba28dfbe7dfa303a9ee139f2f31e4064`
- Patch flags: `USE_PATCH=True`, `USE_INTERPOLATE_PATCH=True`
- Common environment: `MPLBACKEND=Agg`, `MAX_COMPILE_CORE_NUMBER=1`,
  `OMP_NUM_THREADS=1`, `TASK_QUEUE_ENABLE=0`

## Results

| num_levels | Memory policy | Result | Wall time | Max child RSS | Cgroup peak | NPU before/after | Swap | Output |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | 6 GiB RAM, Swap disabled | PASS | 94.069 s | 2700084 KiB | 5394.6 MiB | 5321 / 5323 MiB | 0 / 0 KiB | `seg_pred_num_levels_1.png` |
| 2 | 6 GiB RAM, Swap disabled | OOM / SIGKILL | 56.266 s | 2858220 KiB | 6144 MiB | 5329 / 4796 MiB | 0 / 512 KiB | no new output |
| 4 | 6 GiB RAM, 14 GiB RAM+Swap, swappiness 60 | swap-thrash, terminated | more than 15 min | unavailable after termination | 6144 MiB RAM / 9845 MiB RAM+Swap | 4836 / 4341 MiB | peak observed about 3.64 GiB; post 1792 KiB | no new output |

The level-1 run exited 0 and passed the standard process, cgroup, NPU-memory,
and zero-Swap cleanup gate. Its output SHA256 is
`280f10dceebc57ca05757656981a318be43651b63ccec200b0a4dc1c37a68b4f`.
Against `seg_pred_gpu_golden.png`, the RGBA image has exact-pixel ratio
0.9415598383, MAE 3.7808483180, RMSE 22.7379721198, PSNR 20.995769016 dB,
and maximum channel difference 230. It is not pixel-identical to the golden.

The level-2 child was killed with return code -9. Its cgroup
`memory.memsw.max_usage_in_bytes` reached exactly 6 GiB and
`memory.memsw.failcnt` reached 384, proving the fixed no-Swap limit caused the
failure. `seg_pred.png` retained the level-1 hash and timestamp.

The authorized level-4 exception used a temporary copy of the guard; the
canonical `/home/chenxinji/310B_constraints/run_310b.sh` was not modified. The
run did not hit the 14 GiB combined limit (`memsw_failcnt=0`), but stabilized at
about 5.84 GiB resident plus 3.64 GiB Swap. The main process accumulated only
about 51 CPU seconds after more than 13 minutes and remained blocked in
`trs_logic_cq_recv`; all worker processes were waiting and the output timestamp
did not advance. After an additional bounded observation window, only the owned
process group was terminated. The temporary guard verified NPU cleanup.

After all runs, `/swapfile` was returned to enabled with 0 KiB used, no owned
process or cgroup remained, and `demo.py` was restored to its original SHA256
with `num_levels=4`. The first post-cleanup NPU check reported 4368 MiB and
Health `OK`; the final audit later reported 4392 MiB and Health `Warning`,
consistent with the device's previously observed intermittent `dmp_daemon`
heartbeat warning. No reset was performed.

## Authorized Level-2 Swap Rerun - 2026-08-24

The level-2 configuration was rerun with the same temporary 6 GiB RAM / 14 GiB
RAM+Swap / swappiness 60 policy used for the authorized level-4 experiment. It
completed successfully in 214.14921941 seconds (3 minutes 34.149 seconds) with
child exit code 0. Child user/system CPU time was 38.414528 / 27.293674 seconds,
maximum child RSS was 3566508 KiB, and major/minor page faults were 64639 /
2736168. The cgroup peaks were exactly 6144 MiB RAM and 6933.5 MiB RAM+Swap;
`memory.failcnt` was 232106 and `memory.memsw.failcnt` was 0. NPU memory was
6241 MiB before and 4861 MiB after cleanup. Swap was 0 KiB before, 5120 KiB
after process cleanup, and was then explicitly returned to 0 KiB by restarting
only `/swapfile` under the user's authorization.

The output was saved as `seg_pred_num_levels_2_swap.png`, SHA256
`5a9f504414a63fd84c1c7b74b79546b9c33dd0437962bd32444e4038d4689618`.
Against the fixed GPU golden, its exact-pixel ratio was 0.9475006453, MAE
3.1250828099, RMSE 20.3331874266, PSNR 21.966694331 dB, and maximum channel
difference 230. It is not pixel-identical to the golden. The original
`demo.py` was restored to `num_levels=4` and its fixed SHA256 after the run.

## All Compatibility Patches Disabled - 2026-08-24

The six global monkey patches were disabled with `USE_PATCH=False` and
`USE_INTERPOLATE_PATCH=False`. Incremental full-demo runs found two missing
project integration boundaries:

1. `open_clip/transformer.py` still entered native SDPA through
   `nn.MultiheadAttention`, failing on the missing FP16 FRACTAL_NZ
   `BatchMatMulV2` binary. Its NPU path now explicitly owns Q/K/V projection,
   head reshaping, `scaled_dot_product_attention_310b`, output projection, and
   MLP Linear through `linear_310b`. A standalone `ResidualAttentionBlock`
   CPU/NPU comparison passed with maximum error 0.0019645691.
2. The visual transformer's custom attention output projection still called an
   `nn.Linear` module directly. It now uses the same explicit NPU Linear
   dispatch.
3. `segearth_segmentor.py` still called native bilinear interpolation at five
   segmentation resize boundaries. Those NPU paths now explicitly dispatch to
   `upsample_bilinear2d_310b`; a CPU/NPU comparison passed with maximum error
   0.0042818785.

After these corrections, a level-2 full demo with all patches disabled passed
in 206.491716077 seconds under the authorized 6 GiB RAM / 14 GiB RAM+Swap
policy. Peak RAM was 6144 MiB and peak RAM+Swap was 6911 MiB; NPU memory was
5044 / 4814 MiB before/after cleanup. Its output
`seg_pred_num_levels_2_patch_off.png` has exact-pixel ratio 0.9476024549, MAE
3.1199708194, RMSE 20.3175205373, PSNR 21.9733894603 dB, and maximum channel
difference 230 against the GPU golden. It shares 0.9978161117 exact pixels with
the earlier patch-enabled level-2 output.

The selected final demo configuration is level 1 with all patches disabled.
It passed the standard 6 GiB, zero-Swap guard in 97.110081808 seconds. Maximum
child RSS was 2560496 KiB and cgroup memory/memsw peak was 5964017664 bytes,
with both failure counters zero. NPU memory was 4865 / 4904 MiB before/after
cleanup and Swap remained 0 KiB. Output
`seg_pred_num_levels_1_patch_off.png` has SHA256
`bbb3cef850196c02a6b5b77eca7d6b56dd0d00a85a8a598377f53d61cf2a7f05`.
Against the GPU golden its exact-pixel ratio is 0.9415870831, MAE 3.7811673693,
RMSE 22.7405969278, PSNR 20.9947663990 dB, and maximum channel difference 230.
The project `demo.py` is intentionally retained at level 1 with both patch
flags false.
