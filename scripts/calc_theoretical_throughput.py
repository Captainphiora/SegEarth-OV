"""Calculate theoretical peak throughput for SegEarth-OV on Atlas 200I A2.

Data sources:
  1. profiling_runs/20260901_170804/csv/ — msprof profiling (9 crops, demo_multi 448x448)
  2. eval_acl debug_timing — per-crop breakdown (visual/jbu/numpy)
  3. hardware_environment.md — Atlas 200I A2 (Ascend 310B1) specs

Hardware: Atlas 200I A2 (Ascend 310B1)
  FP16 peak compute: 10 TFLOPS
  Memory bandwidth:  51.2 GB/s (LPDDR4X 96-bit @ 4262 MT/s)
  AI Cores: 1
"""

import csv
import sys

PROF_CSV = "profiling_runs/20260901_170804/csv/op_summary.csv"

PEAK_FP16_TFLOPS = 10.0
PEAK_FP16_FLOPS = PEAK_FP16_TFLOPS * 1e12
PEAK_BW_GBS = 51.2
PEAK_BW_BS = PEAK_BW_GBS * 1e9
RIDGE_POINT = PEAK_FP16_FLOPS / PEAK_BW_BS

PROFILING_NUM_CROPS = 9

VISUAL_MS_PER_CROP = 21.0
JBU_MS_PER_CROP = 3291.0
NUMPY_MS_PER_CROP = 338.0
OVERHEAD_PER_CROP_MS = 2.0
TOTAL_PER_CROP_MS = VISUAL_MS_PER_CROP + JBU_MS_PER_CROP + NUMPY_MS_PER_CROP + OVERHEAD_PER_CROP_MS

ACTUAL_AVG_CROPS = 6.75
ACTUAL_TIME_MS = 24278.0
ACTUAL_TP = 0.04


def parse_shape(s):
    s = s.strip().strip('"')
    if not s or s == 'N/A':
        return ()
    try:
        return tuple(int(x) for x in s.split(',') if x.strip())
    except:
        return ()


def shape_numel(shape):
    r = 1
    for d in shape:
        r *= d
    return r


def dtype_bytes(dtype_str):
    d = dtype_str.strip().upper()
    if 'FLOAT16' in d or 'FP16' in d:
        return 2
    elif 'FLOAT' in d or 'FP32' in d:
        return 4
    elif 'INT8' in d:
        return 1
    elif 'INT32' in d or 'INT64' in d:
        return 4
    return 2


def estimate_mem_traffic(in_shapes, in_dtypes, out_shapes, out_dtypes):
    total = 0
    for shape, dtype in zip(in_shapes, in_dtypes):
        if shape:
            total += shape_numel(shape) * dtype_bytes(dtype)
    for shape, dtype in zip(out_shapes, out_dtypes):
        if shape:
            total += shape_numel(shape) * dtype_bytes(dtype)
    return total


def main():
    total_mem_bytes = 0
    total_device_time_us = 0
    total_wait_time_us = 0
    time_by_type = {}
    mem_by_type = {}
    count_by_type = {}
    mac_time_by_type = {}
    mte_time_by_type = {}

    with open(PROF_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_type = row['OP Type']
            duration = float(row['Task Duration(us)'])
            wait_time = float(row.get('Task Wait Time(us)', 0))
            total_device_time_us += duration
            total_wait_time_us += wait_time

            in_shapes_raw = row.get('Input Shapes', '')
            out_shapes_raw = row.get('Output Shapes', '')
            in_dtypes_raw = row.get('Input Data Types', '')
            out_dtypes_raw = row.get('Output Data Types', '')

            in_shape_strs = [s.strip().strip('"') for s in in_shapes_raw.split(';') if s.strip()]
            out_shape_strs = [s.strip().strip('"') for s in out_shapes_raw.split(';') if s.strip()]
            in_dtype_strs = [s.strip() for s in in_dtypes_raw.split(';') if s.strip()]
            out_dtype_strs = [s.strip() for s in out_dtypes_raw.split(';') if s.strip()]

            in_shapes = [parse_shape(s) for s in in_shape_strs]
            out_shapes = [parse_shape(s) for s in out_shape_strs]

            mem = estimate_mem_traffic(in_shapes, in_dtype_strs, out_shapes, out_dtype_strs)
            total_mem_bytes += mem

            time_by_type[op_type] = time_by_type.get(op_type, 0) + duration
            mem_by_type[op_type] = mem_by_type.get(op_type, 0) + mem
            count_by_type[op_type] = count_by_type.get(op_type, 0) + 1

            mac_r = float(row.get('mac_exe_ratio', 0) or 0)
            aicore_t = float(row.get('aicore_time(us)', 0) or 0)
            mac_time_by_type[op_type] = mac_time_by_type.get(op_type, 0) + mac_r * aicore_t
            mte2_r = float(row.get('mte2_exe_ratio', 0) or 0)
            mte_time_by_type[op_type] = mte_time_by_type.get(op_type, 0) + mte2_r * aicore_t

    total_device_ms = total_device_time_us / 1e3
    total_mem_gb = total_mem_bytes / 1e9

    print("=" * 72)
    print("  SegEarth-OV 理论峰值吞吐量分析")
    print("  Atlas 200I A2 (Ascend 310B1)")
    print("=" * 72)
    print()

    print("【硬件规格】")
    print(f"  FP16 算力:      {PEAK_FP16_TFLOPS} TFLOPS")
    print(f"  内存带宽:       {PEAK_BW_GBS} GB/s (LPDDR4X 96-bit)")
    print(f"  AI Core:        1 个")
    print(f"  Roofline 拐点:  {RIDGE_POINT:.1f} FLOPs/Byte")
    print(f"    (OI > {RIDGE_POINT:.0f} → 算力瓶颈; OI < {RIDGE_POINT:.0f} → 带宽瓶颈)")
    print()

    print("【实测性能 (eval_acl.py, 40 images)】")
    print(f"  平均每图:   {ACTUAL_TIME_MS/1000:.2f}s")
    print(f"  吞吐量:     {ACTUAL_TP} img/s")
    print()

    print("【Per-Crop 实测耗时 (debug_timing)】")
    print(f"  clip_visual.om (NPU):   {VISUAL_MS_PER_CROP:>8.0f} ms ({VISUAL_MS_PER_CROP/TOTAL_PER_CROP_MS*100:.1f}%)")
    print(f"  jbu_upsampler.om (NPU): {JBU_MS_PER_CROP:>8.0f} ms ({JBU_MS_PER_CROP/TOTAL_PER_CROP_MS*100:.1f}%)")
    print(f"  numpy post-proc (CPU):  {NUMPY_MS_PER_CROP:>8.0f} ms ({NUMPY_MS_PER_CROP/TOTAL_PER_CROP_MS*100:.1f}%)")
    print(f"  合计:                   {TOTAL_PER_CROP_MS:>8.0f} ms/crop")
    print(f"  >> JBU 占 {JBU_MS_PER_CROP/TOTAL_PER_CROP_MS*100:.0f}% 的 per-crop 时间，是绝对瓶颈")
    print()

    print("【Profiling 算子分布 ({} crops, device time {:.1f}ms)】".format(
        PROFILING_NUM_CROPS, total_device_ms))
    print(f"  {'算子类型':<28s} {'调用数':>6s} {'耗时ms':>9s} {'占比':>6s} {'MemIO_GB':>9s} {'有效BW':>9s} {'瓶颈':>6s}")
    print(f"  {'-'*28} {'-'*6} {'-'*9} {'-'*6} {'-'*9} {'-'*9} {'-'*6}")

    compute_ops = {'Conv2D', 'MatMulV2', 'BatchMatMulV2'}
    compute_total_ms = 0
    mem_total_ms = 0
    other_total_ms = 0

    for op_type in sorted(time_by_type, key=lambda x: -time_by_type[x]):
        t_ms = time_by_type[op_type] / 1e3
        c = count_by_type[op_type]
        m_gb = mem_by_type.get(op_type, 0) / 1e9
        ratio = t_ms / total_device_ms * 100
        if t_ms > 0 and m_gb > 0:
            eff_bw = m_gb / (t_ms / 1000)
        else:
            eff_bw = 0
        mac_t = mac_time_by_type.get(op_type, 0) / 1e3
        mte_t = mte_time_by_type.get(op_type, 0) / 1e3
        if mac_t > mte_t and mac_t > 0.01:
            bound = "COMP"
        else:
            bound = "MEM"
        if ratio < 0.01:
            continue
        print(f"  {op_type:<28s} {c:>6d} {t_ms:>9.1f} {ratio:>5.1f}% {m_gb:>9.2f} {eff_bw:>7.1f}G/s {bound:>6s}")

        if op_type in compute_ops:
            compute_total_ms += t_ms
        elif bound == "MEM":
            mem_total_ms += t_ms
        else:
            other_total_ms += t_ms

    print()
    eff_bw_overall = total_mem_gb / (total_device_ms / 1000)
    print(f"  整体有效带宽: {eff_bw_overall:.1f} GB/s ({eff_bw_overall/PEAK_BW_GBS*100:.0f}% of peak {PEAK_BW_GBS} GB/s)")
    print()

    print("=" * 72)
    print("【理论峰值吞吐量计算】")
    print("=" * 72)
    print()

    mem_per_crop_gb = total_mem_gb / PROFILING_NUM_CROPS
    mem_per_crop_ms_peak = mem_per_crop_gb / PEAK_BW_GBS * 1000

    print(f"━━━ 计算方法：Roofline 模型 ━━━")
    print()
    print(f"  核心公式:")
    print(f"    Operational Intensity (OI) = FLOPs / Bytes")
    print(f"    Ridge Point = Peak_FLOPS / Peak_BW = {PEAK_FP16_TFLOPS}T / {PEAK_BW_GBS}G = {RIDGE_POINT:.0f} FLOPs/Byte")
    print()
    print(f"    若 OI > Ridge Point → 算力瓶颈:  T_理论 = FLOPs / Peak_FLOPS")
    print(f"    若 OI < Ridge Point → 带宽瓶颈:  T_理论 = Bytes / Peak_BW")
    print()
    print(f"  本工作负载特征:")
    print(f"    - 绝大多数算子 OI << {RIDGE_POINT:.0f}（StridedSliceD=0, TransData=0, Mul=0）")
    print(f"    - 仅 Conv2D 的 OI ≈ 218 略高于拐点")
    print(f"    - 整体为 Memory-Bound 工作负载")
    print()

    print(f"━━━ 场景 A: 纯带宽瓶颈理论峰值（NPU 所有算子达峰值带宽）━━━")
    print()
    print(f"  每 crop 内存流量 (profiling 估算):  {mem_per_crop_gb:.1f} GB")
    print(f"  每 crop 理论最短 NPU 时间:          {mem_per_crop_gb:.1f} GB / {PEAK_BW_GBS} GB/s = {mem_per_crop_ms_peak:.0f} ms")
    npu_per_crop_a = mem_per_crop_ms_peak
    total_per_crop_a = npu_per_crop_a + NUMPY_MS_PER_CROP
    for n_crops in [6, 9]:
        t_img = n_crops * total_per_crop_a
        tp = 1000.0 / t_img
        print(f"  {n_crops} crops: {n_crops} × ({npu_per_crop_a:.0f} + {NUMPY_MS_PER_CROP:.0f})ms = {t_img:.0f}ms → {tp:.2f} img/s")
    print(f"  >> 这是绝对上限，假设 100% 带宽利用率（实际不可达）")
    print()

    print(f"━━━ 场景 B: 70% 带宽利用率（优化后现实目标）━━━")
    print()
    bw_util_b = 0.70
    eff_bw_b = PEAK_BW_GBS * bw_util_b
    npu_per_crop_b = mem_per_crop_gb / eff_bw_b * 1000
    total_per_crop_b = npu_per_crop_b + NUMPY_MS_PER_CROP
    for n_crops in [6, 9]:
        t_img = n_crops * total_per_crop_b
        tp = 1000.0 / t_img
        print(f"  {n_crops} crops: {n_crops} × ({npu_per_crop_b:.0f} + {NUMPY_MS_PER_CROP:.0f})ms = {t_img:.0f}ms → {tp:.3f} img/s")
    print(f"  >> 考虑实际内存访问效率，70% 是优化后的合理目标")
    print()

    print(f"━━━ 场景 C: 当前实际性能 (38.6% 带宽利用率) ━━━")
    print()
    bw_util_c = eff_bw_overall / PEAK_BW_GBS
    npu_per_crop_c = JBU_MS_PER_CROP + VISUAL_MS_PER_CROP
    total_per_crop_c = npu_per_crop_c + NUMPY_MS_PER_CROP
    for n_crops in [6, 9]:
        t_img = n_crops * total_per_crop_c
        tp = 1000.0 / t_img
        print(f"  {n_crops} crops: {n_crops} × ({npu_per_crop_c:.0f} + {NUMPY_MS_PER_CROP:.0f})ms = {t_img:.0f}ms → {tp:.3f} img/s")
    print(f"  >> 当前带宽利用率仅 {bw_util_c*100:.0f}%，主要受 StridedSliceD 拖累")
    print()

    print("=" * 72)
    print("【瓶颈分析与计算方式差异】")
    print("=" * 72)
    print()

    print("1. Compute-Bound 工作负载的吞吐量公式:")
    print("   Throughput = Peak_FLOPS / (FLOPs_per_image)")
    print(f"   本项目如果 compute-bound: 10T / ~1017G = ~9.8 img/s")
    print(f"   但实际远达不到，因为本工作负载不是 compute-bound。")
    print()

    print("2. Memory-Bound 工作负载的吞吐量公式 (本项目适用):")
    print("   Throughput = Peak_BW / (Bytes_per_image)")
    print(f"   本项目: {PEAK_BW_GBS} GB/s / {total_mem_gb:.0f} GB = {PEAK_BW_GBS/total_mem_gb:.3f} img/s")
    print(f"   这是 NPU 端的绝对理论上限。")
    print()

    print("3. Host-Bound 工作负载的吞吐量公式:")
    print("   Throughput = 1 / (T_host_per_image)")
    print(f"   本项目 CPU 部分 (numpy): {NUMPY_MS_PER_CROP:.0f}ms × 6~9 crops = {NUMPY_MS_PER_CROP*6:.0f}~{NUMPY_MS_PER_CROP*9:.0f}ms")
    print(f"   如果 NPU 时间优化到趋近于 0，CPU 仍是 {1000/(NUMPY_MS_PER_CROP*7.5):.1f} img/s 的硬限")
    print()

    print("4. 为什么不同瓶颈计算方式不同？")
    print("   - Compute-bound: 瓶颈是 AI Core 的 MAC 单元 → 看总 FLOPs")
    print("   - Memory-bound:  瓶颈是 DRAM ↔ AI Core 的数据搬运 → 看总 Bytes")
    print("   - Host-bound:    瓶颈是 CPU 计算或 host→device 调度 → 看 CPU 耗时")
    print("   实际系统通常是三者的叠加，取耗时最长的一个作为主瓶颈。")
    print()

    print("5. 本项目的具体瓶颈链:")
    print(f"   JBU upsampler (NPU, {JBU_MS_PER_CROP/TOTAL_PER_CROP_MS*100:.0f}% per crop)")
    print(f"     └─ StridedSliceD ({time_by_type.get('StridedSliceD',0)/total_device_time_us*100:.0f}% of device time) → 纯内存搬运，0 FLOPs")
    print(f"     └─ TransData ({time_by_type.get('TransData',0)/total_device_time_us*100:.0f}%) → 数据格式转换 (NCHW↔NC1HWC0)")
    print(f"     └─ Mul + Fusion ({(time_by_type.get('Mul',0)+time_by_type.get('AutomaticBufferFusionOp',0))/total_device_time_us*100:.0f}%) → 逐元素运算")
    print(f"     └─ Conv2D ({time_by_type.get('Conv2D',0)/total_device_time_us*100:.0f}%) → 唯一的计算密集型算子")
    print()

    print("=" * 72)
    print("【汇总】")
    print("=" * 72)
    print()
    print(f"  {'场景':<38s} {'每图耗时':>10s} {'吞吐量':>12s} {'倍率':>8s}")
    print(f"  {'-'*38} {'-'*10} {'-'*12} {'-'*8}")

    rows = [
        ("当前实测 (40 images avg)", ACTUAL_TIME_MS, ACTUAL_TP),
    ]
    for n_crops in [6, 9]:
        t_c = n_crops * total_per_crop_c
        rows.append((f"  当前性能 ({n_crops} crops)", t_c, 1000/t_c))
    for n_crops in [6, 9]:
        t_b = n_crops * total_per_crop_b
        rows.append((f"  优化目标 70%BW ({n_crops} crops)", t_b, 1000/t_b))
    for n_crops in [6, 9]:
        t_a = n_crops * total_per_crop_a
        rows.append((f"  理论峰值 100%BW ({n_crops} crops)", t_a, 1000/t_a))

    cpu_only_6 = 6 * NUMPY_MS_PER_CROP
    rows.append(("  CPU硬限 (NPU=0, 6crops)", cpu_only_6, 1000/cpu_only_6))

    for name, t, tp in rows:
        speedup = f"{ACTUAL_TIME_MS/t:.1f}x" if t > 0 else "—"
        print(f"  {name:<38s} {t/1000:>8.2f}s {tp:>10.3f}/s {speedup:>8s}")
    print()


if __name__ == "__main__":
    main()
