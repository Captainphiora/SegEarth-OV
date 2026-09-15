# SegEarth-OV 遥感语义分割 理论峰值吞吐量分析报告

> 平台: Atlas 200I A2 (Ascend 310B1)  
> 日期: 2026-09-02  
> Profiling 来源: `profiling_runs/eval_acl_1img_20260902_021328/` (eval_acl.py, 1 image, 6 crops)  
> Benchmark 来源: `work_logs/eval_acl_20260902_014116/eval.log` (40 images)

---

## 1. 结论

| 指标 | 值 |
|------|-----|
| **实测吞吐量** | **0.04 img/s** (24.28s/image) |
| **理论峰值吞吐量 (Memory-Bound)** | **0.135 img/s** (NPU only) / **0.106 img/s** (含 CPU) |
| **理论峰值吞吐量 (Compute-Bound, 假设)** | **14.75 img/s** (NPU only) / **0.48 img/s** (含 CPU) |
| 实测 vs 理论峰值 (Memory-Bound) | 距离峰值 **2.6x** |
| 当前瓶颈类型 | **Memory-Bound** (内存带宽瓶颈) |
| 绝对瓶颈算子 | JBU Upsampler (占 per-crop 时间 90%) |

**本工作负载在 310B1 上是典型的 Memory-Bound 场景**, 因此理论峰值吞吐量应使用带宽公式计算,
结果为 **0.106 img/s (含 CPU 时间)** 或 **0.135 img/s (纯 NPU)**。

---

## 2. 硬件规格

| 参数 | 值 | 说明 |
|------|-----|------|
| NPU 芯片 | Ascend 310B1 | Atlas 200I A2 开发者套件 |
| FP16 算力 | 10 TFLOPS | 1 个 AI Core |
| 内存带宽 | 51.2 GB/s | LPDDR4X 96-bit @ 4262 MT/s |
| 内存容量 | 12 GB | CPU 和 NPU 共享 |
| Roofline 拐点 | 195.3 FLOPs/Byte | = 10T ÷ 51.2G |

**Roofline 拐点的含义**: 当一个算子的"算术强度"（每字节数据对应的浮点运算次数）超过 195.3 时,
该算子是算力瓶颈 (Compute-Bound); 低于 195.3 时, 该算子是带宽瓶颈 (Memory-Bound)。

---

## 3. 推理流程与 Per-Crop 耗时

### 3.1 推理流程

```
输入图像 (448×448)
    │
    ▼
Slide Window 切割 → 6 个 224×224 crops (stride=112, keep_ratio=True)
    │
    ▼ ×6 (每个 crop)
┌────────────────────────────────┐
│ ① clip_visual.om  (NPU)       │  提取 patch tokens [1,196,512] + cls_token
│ ② reshape + transpose (CPU)   │  [1,196,512] → [1,512,14,14]
│ ③ jbu_upsampler.om (NPU)      │  [1,512,14,14] + 原图 → [1,512,224,224]
│ ④ L2 norm + einsum  (CPU)     │  计算 per-pixel class logits
└────────────────────────────────┘
    │
    ▼
合并 6 个 crop 的 logits → softmax → argmax → 分割结果
```

### 3.2 Per-Crop 实测耗时

来自 `--debug-timing` 实测 (非 profiling, 无额外开销):

| 阶段 | 耗时 | 占比 | 执行位置 |
|------|------|------|----------|
| clip_visual.om | 21 ms | 0.6% | NPU |
| jbu_upsampler.om | 3,291 ms | 90.1% | NPU |
| numpy 后处理 | 338 ms | 9.3% | CPU |
| **合计** | **3,652 ms** | 100% | |

来自 profiling step_trace (新采集):

| 模型 | Device Time | 调用数 |
|------|------------|--------|
| clip_visual.om (Model 1) | 19.8 ms/crop | 6 |
| jbu_upsampler.om (Model 2) | 3,170 ms/crop | 6 |

**结论: JBU upsampler 占 per-crop 时间的 90%, 是绝对瓶颈。**

---

## 4. Profiling 算子热点分析

### 4.1 算子耗时分布 (6 crops, 总 device time = 19,132 ms)

| 算子类型 | 调用数 | 耗时 (ms) | 占比 | 属于 | 瓶颈类型 |
|---------|--------|-----------|------|------|----------|
| StridedSliceD | 9,288 | 7,048 | 36.8% | JBU | MEM (OI=0) |
| Conv2D | 228 | 3,914 | 20.5% | JBU+CLIP | MEM* |
| Mul | 1,224 | 2,359 | 12.3% | JBU | MEM (OI≈0) |
| TransData | 1,302 | 2,266 | 11.8% | JBU | MEM (OI=0) |
| AutomaticBufferFusionOp | 210 | 1,672 | 8.7% | JBU | MEM |
| ConcatD | 150 | 636 | 3.3% | JBU | MEM (OI=0) |
| Transpose | 492 | 504 | 2.6% | JBU+CLIP | MEM (OI=0) |
| PadV3 | 144 | 467 | 2.4% | JBU | MEM |
| MatMulV2 | 276 | 51 | 0.3% | CLIP | MEM (OI=98.5) |
| BatchMatMulV2 | 168 | 5 | 0.03% | CLIP | MEM (OI=6.1) |

> *Conv2D 的 OI ≈ 218, 略超拐点 195, 理论上是 Compute-Bound, 但 profiling 显示
> 其 MTE (内存搬运) 管线利用率 > MAC (计算) 管线, 实际仍受带宽限制。

### 4.2 JBU 中最耗时的算子及其 shape

| 算子 | Shape | 每 crop 耗时 | 说明 |
|------|-------|-------------|------|
| Conv2D | [1,32,224,224,16] | 539 ms | 全分辨率 512-ch 卷积 |
| TransData | [1,512,224,224] | 361 ms | NCHW↔NC1HWC0 格式转换 |
| Mul | [1,512,224,224] | 308 ms | 逐元素乘 |
| FusionOp | [1,32,224,224,16]×4 | 189 ms | 融合算子 |
| Transpose | [1,32,121,224,224] | 60 ms | 大张量转置 |
| StridedSliceD | up_3/Slice_* | 1,175 ms | 各种 Slice 操作 |

**关键发现**: JBU 在 224×224 分辨率上操作 512 通道的特征图, 每个 Slice/Mul/TransData
都需要搬运 ~100MB 的数据, 是典型的 "大张量低算术强度" 内存密集型模式。

### 4.3 关键指标

| 指标 | 值 | 说明 |
|------|-----|------|
| 总 FLOPs (6 crops) | 678 GFLOPs | 以 Conv2D + MatMul + BMM 估算 |
| 总内存流量 (6 crops) | 378 GB | 所有算子输入输出张量大小之和 |
| 有效带宽 | 19.8 GB/s | = 378 GB / 19.1s |
| 带宽利用率 | **38.6%** | = 19.8 / 51.2 |
| 有效算力 | 35.4 GFLOPS | = 678G / 19.1s |
| 算力利用率 | **0.35%** | = 35.4G / 10T |

算力利用率极低 (0.35%), 而带宽利用率为 38.6%, 说明 **NPU 的 MAC 单元绝大多数时间
在等数据**, 而非在做计算。这是 Memory-Bound 的典型表现。

---

## 5. 理论峰值吞吐量计算

### 5.1 核心方法: Roofline 模型

Roofline 模型根据工作负载的**算术强度 (Operational Intensity, OI)** 判断瓶颈类型,
然后选择对应的公式计算理论峰值:

```
OI = FLOPs / Bytes    (每字节数据需要多少次浮点运算)

Ridge Point = Peak_FLOPS / Peak_BW = 10 TFLOPS / 51.2 GB/s = 195.3 FLOPs/Byte

若 OI ≥ Ridge Point → Compute-Bound:  T_理论 = Total_FLOPs / Peak_FLOPS
若 OI < Ridge Point → Memory-Bound:   T_理论 = Total_Bytes / Peak_BW
```

### 5.2 本工作负载的算术强度

```
OI_整体 = 678 GFLOPs / 378 GB = 1.79 FLOPs/Byte
```

**1.79 远远小于拐点 195.3**, 因此本工作负载整体是 **Memory-Bound**。

直觉理解: 每从内存搬运 1 字节数据, NPU 只做了 1.79 次浮点运算, 但 310B1 的硬件能力是
每搬 1 字节就能做 195.3 次运算。也就是说, 绝大部分时间硬件的计算单元都在空等数据。

### 5.3 Memory-Bound 理论峰值 (本工作负载适用)

```
T_NPU = Total_Bytes / Peak_BW
      = 378 GB / 51.2 GB/s
      = 7,388 ms

T_CPU = 6 crops × 338 ms = 2,028 ms    (numpy L2-norm + einsum, 无法省略)

T_total = T_NPU + T_CPU = 9,416 ms     (NPU 与 CPU 串行, 无法重叠)

理论峰值吞吐量 = 1 / 9.416s = 0.106 img/s
```

> 这里 T_NPU 假设所有内存操作都以**峰值带宽 51.2 GB/s** 运行, 实际不可达
> (非连续访存、kernel launch 开销等), 因此 0.106 img/s 是**绝对上限**。

如果不考虑 CPU 时间 (假设 CPU 部分移到 NPU 或完全重叠):

```
理论峰值 (纯 NPU) = 1 / 7.388s = 0.135 img/s
```

### 5.4 Compute-Bound 理论峰值 (假设场景)

如果本工作负载是 Compute-Bound (即内存带宽无限大):

```
T_NPU = Total_FLOPs / Peak_FLOPS
      = 678 GFLOPs / 10 TFLOPS
      = 67.8 ms

T_CPU = 2,028 ms

T_total = 67.8 + 2,028 = 2,096 ms

理论峰值 (Compute-Bound) = 1 / 2.096s = 0.477 img/s
```

纯 NPU (不含 CPU):

```
理论峰值 (纯 NPU, Compute-Bound) = 1 / 0.0678s = 14.75 img/s
```

> 注意: Compute-Bound 假设不适用于本工作负载, 仅作为对比参考。
> 要让本工作负载变为 Compute-Bound, 需要将 OI 从 1.79 提升到 195+,
> 这意味着要将内存流量减少 ~100 倍 (通过算子融合、避免冗余搬运等),
> 在当前模型架构下不现实。

### 5.5 现实优化目标

| 场景 | 带宽利用率 | 每图耗时 | 吞吐量 | vs 实测 |
|------|-----------|---------|--------|---------|
| 当前实测 | 38.6% | 24.28s | 0.04 img/s | 1.0x |
| 优化目标 (70% BW) | 70% | 12.6s | 0.08 img/s | 1.9x |
| 理论峰值 (100% BW) | 100% | 9.4s | 0.106 img/s | 2.6x |
| Compute-Bound 假设 | N/A | 2.1s | 0.48 img/s | 11.5x |

---

## 6. 不同瓶颈类型的计算方式对比

### 为什么不同 bottleneck 要用不同公式？

NPU 内部有两条关键流水线:
- **MAC (矩阵计算单元)**: 执行卷积、矩阵乘等计算
- **MTE (内存搬运引擎)**: 在 DRAM 和片上缓存之间搬运数据

两条流水线并行工作, 但谁更慢谁就成为瓶颈:

| 瓶颈类型 | 公式 | 看什么 | 适用场景 |
|---------|------|--------|---------|
| **Compute-Bound** | `T = FLOPs / Peak_FLOPS` | 总浮点运算量 | 大矩阵乘、大 batch 卷积 (OI 高) |
| **Memory-Bound** | `T = Bytes / Peak_BW` | 总数据搬运量 | Slice、Transpose、小 batch 卷积 (OI 低) |
| **Host-Bound** | `T = T_cpu` | CPU 端耗时 | CPU 后处理重、host dispatch 慢 |

```
                    Performance
                        ▲
                        │          ┌─────────── Peak FLOPS = 10 TFLOPS
                        │         ╱
                        │        ╱  Compute-Bound
                        │       ╱   (T = FLOPs/Peak)
                        │      ╱
   Memory-Bound        │     ╱
   (T = Bytes/BW)      │    ╱
                        │   ╱
                        │  ╱
                        │ ╱
                        │╱
                        └──────────────────────► OI (FLOPs/Byte)
                             ▲
                             │
                        Ridge Point
                        = 195.3 F/B

    本工作负载 OI = 1.79 ← 远在左侧, Memory-Bound
```

### 本项目为什么是 Memory-Bound？

1. **JBU upsampler 的主要操作是数据搬运**: StridedSliceD (36.8%) + TransData (11.8%) +
   Mul (12.3%) 的 OI 都接近 0, 它们本质上就是在内存中移动数据
2. **Conv2D 的 batch size = 1**: batch=1 时, 每次卷积需要重新从 DRAM 加载权重,
   无法摊薄权重加载开销, OI 被压低
3. **310B1 的算力/带宽比极高 (195:1)**: 相比于高端 GPU (如 A100 的 ~310:1),
   310B1 的拐点就很高, 很多操作都落入 Memory-Bound 区域

### 如果要让本工作负载变成 Compute-Bound

需要将 OI 从 1.79 提升到 195 以上, 可能的途径:
- **算子融合**: 将 Slice + Conv + TransData 融合为一个 kernel, 避免中间结果写回 DRAM
- **增大 batch size**: 多个 crop 同时计算 (但受限于 12GB 内存)
- **减少数据搬运**: 优化 JBU 的网络结构, 减少不必要的 Slice/Transpose

---

## 7. 实测与理论的差距分析

| 来源 | 耗时 | 说明 |
|------|------|------|
| 理论峰值 (Memory-Bound, NPU+CPU) | 9,416 ms | 100% 带宽利用率 |
| Profiling device time (NPU only) | 19,132 ms | 38.6% 带宽利用率 |
| 实测总时间 (wall clock) | 24,044 ms | 含所有开销 |

差距来源:

| 因素 | 估算耗时 | 说明 |
|------|---------|------|
| NPU 算子执行 (device) | 19,132 ms | Profiling 测得 |
| CPU numpy 后处理 | 2,028 ms | 6 × 338ms |
| Host dispatch + memcpy | ~2,900 ms | 24,044 - 19,132 - 2,028 |
| → 其中 NPU 带宽未打满 | ~11,744 ms | = 19,132 - 7,388 理论值 |

**最大的优化空间在于 NPU 带宽利用率**: 当前仅 38.6%, 主要被 StridedSliceD
(有效 BW 仅 18.6 GB/s) 和大量 TransData 格式转换拖累。

---

## 8. 附录: 数据来源与复现

### Profiling 采集

```bash
# 1. 采集 (msprof)
bash /home/chenxinji/npu-tools/profiling/npu-profile.sh \
  "python -u scripts/eval_acl.py --config configs/cfg_udd5.py --backend acl \
   --om-dir models/om --device-id 0 --template full --max-samples 1" \
  --label eval_acl_1img

# 2. 解析二进制 → sqlite (CANN 9.0.0 Python tool)
cd /usr/local/Ascend/cann-9.0.0/tools/profiler/profiler_tool/analysis
python -c "
import sys; sys.argv = ['msprof', 'import', '-dir', '<PROF_DIR>']
from msinterface.msprof_entrance import MsprofEntrance; MsprofEntrance().main()
"

# 3. 导出 sqlite → CSV
python -c "
import sys; sys.argv = ['msprof', 'export', 'summary', '-dir', '<PROF_DIR>', '--format', 'csv']
from msinterface.msprof_entrance import MsprofEntrance; MsprofEntrance().main()
"
```

### Per-crop timing

```bash
python -u scripts/eval_acl.py --config configs/cfg_udd5.py --backend acl \
  --om-dir models/om --device-id 0 --template full --max-samples 1 \
  --debug-timing
```

### 理论峰值计算

```bash
python scripts/calc_theoretical_throughput.py
```
