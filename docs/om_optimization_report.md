# SegEarth-OV NPU 适配与 OM 推理优化技术文档

## 1. 项目概览

SegEarth-OV 是一个基于 CLIP ViT-B/16 + JBU upsampler 的开放词汇遥感语义分割模型。本文档记录将该模型从 PyTorch 适配到昇腾 NPU 的完整技术路线，包括 ONNX 导出、ATC 编译、OM 推理优化，以及最终性能结果。

**硬件环境**：Ascend 910_9382 (64GB HBM), CANN 9.0.0, PyTorch 2.8.0 + torch_npu 2.8.0.post4

**模型结构**：
```
输入图像 448×448 → 9 个 224×224 crop (stride=112 sliding window)
每个 crop:
  1. CLIP Visual Encoder: [1,3,224,224] → cls_token [1,512] + patch_tokens [1,196,512]
  2. JBU Upsampler (4级): [1,512,14,14] → [1,512,224,224]  (14→28→56→112→224)
  3. Logits: image_features × query_features → [1,nq,224,224]
合并 9 crop → 最终分割 [448,448]
```

## 2. PyTorch → ONNX → OM 导出路线

### 2.1 ONNX 导出 (`scripts/export_onnx.py`)

三个独立 ONNX 模型：

| 模型 | 输入 | 输出 | 导出要点 |
|------|------|------|---------|
| `clip_visual.onnx` | image [1,3,224,224] fp16 | cls_token [1,512], patch_tokens [1,196,512] | LayerNorm patch 避免 fp32 cast |
| `clip_text.onnx` | text [N,77] int64 | text_features [N,512] fp16 | 动态 batch axis |
| `jbu_upsampler.onnx` | source [1,512,14,14] + guidance [1,3,224,224] | upsampled [1,512,224,224] | 多项算子替换（见下文）|

**JBU 导出的关键适配**：

1. **bicubic 上采样** → depthwise conv2d 等价实现（ATC 不支持 `Resize` 的 `half_pixel` 坐标模式）
2. **`torch.nn.Unfold`** → 逐位置 slice 累加（ONNX 无 Im2Col 算子，导出为 Gather → ATC 映射为 GatherV2，极慢）
3. **`adaptive_avg_pool2d`** → 固定尺寸 `_upsample_step`（导出时用固定 shape 替代动态 pool）
4. **bicubic conv kernel** → `register_buffer` 注册（不能在 forward 中动态创建）
5. **`_inject_bicubic_buffers()`** 公共函数确保 combined 和 per-stage 导出共用 buffer 注入逻辑

### 2.2 ATC 编译

```bash
source scripts/env_npu.sh
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.0.0/opp
atc --model=models/onnx/xxx.onnx --framework=5 --output=models/om/xxx \
    --soc_version=Ascend910_9382 --log=error
```

- SoC 版本通过 `torch_npu.npu.get_device_name()` 确认
- 必须手动设置 `ASCEND_OPP_PATH`
- 编译产物存放在 `models/om/`

### 2.3 产物清单

```
models/onnx/
├── clip_visual.onnx / .om     — Visual encoder
├── clip_text.onnx / .om       — Text encoder
├── jbu_upsampler.onnx / .om   — JBU combined (4 stage + fixup)
├── jbu_stage{1-4}.onnx / .om  — JBU per-stage (可选)
└── jbu_fixup.onnx / .om       — JBU fixup projection (可选)
```

## 3. 优化手段与演进

### 3.1 GatherV2 → Slice 累加 (`6d83b0e`, `a8a5098`)

**问题**：`torch.nn.Unfold` 导出为 ONNX `Gather` → ATC 编译为 `GatherV2`，单 crop 占 98.6%（3978ms）。

**方案**：用 K² 次 tensor slice + 逐位置累加替代 Unfold，避免 Gather 和大中间张量：
```python
result = input[:,:,0:h2,0:w2] * filters[:,0:1,:,:]
for idx in range(1, K*K):
    i, j = idx // K, idx % K
    result = result + input[:,:,i:i+h2,j:j+w2] * filters[:,idx:idx+1,:,:]
```

**效果**：38.4s → 2.1s（18× 加速）

### 3.2 Buffer 预分配 (`627efe2`)

**问题**：`_OmModel.__call__` 每次调用 `acl.rt.malloc` + `acl.rt.free`，18 次 OM 调用累积开销大。

**方案**：device/host memory 在 `__init__` 时预分配，`__call__` 只做 memcpy + execute。dataset 对象每次重建（ACL 不支持跨 execute 复用）。

### 3.3 Logits NPU 化 (`1b656ea`)

**问题**：logits 计算用 numpy fp32 einsum 在 CPU 上，占 31%（1.38s/9crops）。

**方案**：normalize + einsum + cls_token 修正全部移到 NPU torch fp32。

**关键发现**：torch_npu 操作会切换 ACL 活跃 context，导致后续 OM execute 输出全零。修复：`_OmModel.__call__` 每次调用前 `acl.rt.set_context()` 恢复。

### 3.4 Im2col ONNX Parser 插件 (`23981c1`, `bf59a87`)

**目标**：让 OM 直接走 CANN 内置 Im2col 算子（与 PyTorch 的 aclnnIm2col 同源）。

**已完成**：
- C++ ONNX parser 插件 (`custom_op/im2col_onnx_plugin/im2col_plugin.cc`)
- InferShape 函数 (`im2col_infershape.cc`)
- 简单模型全链路验证通过
- CANN Im2col 输出 4D `[N, C*K², outH, outW]`，加 `Flatten(axis=2)` 转为 PyTorch 3D

**未解决**：复杂 JBU 图中 GE InferShape 无法获取 Im2col 输入 shape（CANN 框架限制，自定义 optiling .so 不从 ASCEND_CUSTOM_OPP_PATH 加载）。

### 3.5 混合后端 (`530c435`)

**思路**：取各组件最优实现——Visual OM 比 PyTorch 快 15×，JBU PyTorch 比 OM 快 50×。

| 组件 | 实现 | 原因 |
|------|------|------|
| Visual Encoder | OM | ATC 图优化，3.5ms/crop vs PyTorch 50ms |
| Text Encoder | OM | 同上 |
| JBU Upsampler | PyTorch (NPU) | 原生 aclnnIm2col，4ms/crop vs OM 115ms |
| Logits | PyTorch (NPU) | fp32 matmul on NPU，0.1ms/crop |

## 4. 推理后端一览

`utils/session.py` 实现了统一的 `Session.from_config(cfg)` 接口：

| `session_type` | 类 | 说明 |
|---|---|---|
| `pytorch` | `PyTorchSession` | 全 PyTorch NPU，精度基线 |
| `acl` | `AclSession` | 全 OM 推理 (combined JBU) |
| `acl_hybrid` | `AclHybridSession` | Visual/Text OM + JBU PyTorch |
| `acl_stage` | `AclStageSession` | 全 OM 推理 (per-stage JBU) |

## 5. 性能与显存对比

测试条件：448×448 图像，9 个 224×224 crop，warmup=1，repeat=3，独立子进程隔离。

| 方案 | 推理时间 | HBM Δ(load) | HBM Δ(infer) | 精度 (vs PyTorch) |
|------|:---:|:---:|:---:|:---:|
| **PyTorch** | **0.74s** | +706 MB | +16,612 MB | baseline |
| **ACL Hybrid** | **0.83s** | +388 MB | +16,210 MB | 97.2% |
| ACL (combined) | 1.97s | +6,659 MB | +7,066 MB | 97.2% |
| ACL (per-stage) | 6.42s | +16,442 MB | +16,866 MB | 97.0% |

**关键结论**：
- ACL Hybrid 最接近 PyTorch（仅差 12%），且模型加载显存最低（+388 MB）
- 纯 ACL 推理时 HBM 峰值最低（+7,066 MB），适合显存受限场景
- ACL per-stage 无优势（多次 OM 调用开销 + PIL guidance 下采样）

## 6. 复现指南

### 6.1 环境准备

```bash
# 激活环境（自动设置 conda + CANN + NPU）
source scripts/env_npu.sh
```

### 6.2 ONNX 导出 + ATC 编译

```bash
# 导出全部 ONNX（visual, text, jbu_combined, jbu_stages）
python scripts/export_onnx.py

# ATC 编译（Ascend 910_9382）
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.0.0/opp
for model in clip_visual clip_text jbu_upsampler; do
    atc --model=models/onnx/${model}.onnx --framework=5 \
        --output=models/om/${model} \
        --soc_version=Ascend910_9382 --log=error
done
```

### 6.3 运行 Demo

```bash
# PyTorch baseline
python scripts/demo_multi.py --backend pytorch --size 448

# ACL 纯 OM (combined JBU)
python scripts/demo_multi.py --backend acl --size 448

# ACL 混合后端 (Visual OM + JBU PyTorch) — 推荐
python scripts/demo_multi.py --backend acl_hybrid --size 448
```

输出保存到 `results/<backend>/seg_pred.png`。

### 6.4 运行 Benchmark

```bash
# 全部四个方案对比（子进程隔离，自动记录 JSON + log）
python scripts/benchmark_om.py --backends pytorch acl acl_hybrid acl_stage

# 只跑指定方案
python scripts/benchmark_om.py --backends acl_hybrid --warmup 1 --repeat 5
```

日志保存到 `work_logs/benchmark_YYYYMMDD_HHMMSS.{log,json}`。

### 6.5 ACL Profiling

```bash
# 粗粒度 Python timer + msprof 算子级采集
python scripts/profile_acl.py

# 解析 msprof 数据
msprof --export=on --output=profiling_output/acl_profile
```

## 7. 关键文件索引

| 文件 | 作用 |
|------|------|
| `scripts/export_onnx.py` | ONNX 导出（含 Unfold→Slice 替换、bicubic conv 等适配） |
| `scripts/demo_multi.py` | 多后端统一 demo 入口 |
| `scripts/benchmark_om.py` | 综合 benchmark（子进程隔离、HBM 监控、精度对比、日志导出） |
| `scripts/profile_acl.py` | ACL profiling（Python timer + msprof 采集） |
| `utils/session.py` | 多后端 Session 管理器（PyTorch / ACL / Hybrid / Stage） |
| `custom_op/im2col_onnx_plugin/` | Im2col ONNX parser 插件（C++ .so）+ InferShape |
| `docs/profiling_analysis.md` | 详细 profiling 分析报告（含优化前后算子对比） |
| `docs/pytorch_to_om_guide.md` | PyTorch→ONNX→OM 全流程转换指南 |

## 8. 已知限制与后续方向

1. **Im2col 原生算子路径未完整打通**：ONNX parser 插件已写好，简单模型验证通过，但复杂 JBU 图中 GE InferShape 无法获取 Im2col 输入 shape。需要 CANN 自定义算子完整工程（op_host DSL）或 CANN 版本升级。

2. **ACL context 与 torch_npu 冲突**：torch_npu 操作会切换 ACL 活跃 context。已在 `_OmModel.__call__` 中加 `set_context` 修复，但退出时 torch_npu cleanup 会报 context null warning（无害）。

3. **310B1 移植**：已完成。详见下方 310B1 适配章节。

## 9. 310B1 适配

### 9.1 ONNX 预处理

310B1 算子库与 910 存在差异，直接编译会出现以下问题：

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|---------|
| `Mod` 常量算子 | 产物带 `_linux_aarch64` 后缀、体积翻倍、推理极慢 | PyTorch opset17 导出的 attention 层含 `Mod(const, const)` 残留，310B1 无高优实现走 fallback | `onnx-simplifier` 常量折叠消除 |
| `ArgMax` 算子 | `Performance_Not_Optimal_Operator` 警告 | `text.argmax(dim=-1)` 导出为 ArgMax，310B1 无高优实现（910 有） | `_argmax_no_argmax()`: `ReduceMax + Equal + Where + ReduceMax` 等价替代 |

**导出流程**（`scripts/export_onnx.py`）：

```bash
# 1. 导出 ONNX（已内置 ArgMax 替代）
python scripts/export_onnx.py

# 2. onnx-simplifier 消除 Mod 等常量算子
python -c "
import onnx; from onnxsim import simplify
for name in ['clip_visual', 'clip_text']:
    m = onnx.load(f'models/onnx/{name}.onnx')
    m_sim, _ = simplify(m)
    onnx.save(m_sim, f'models/onnx/{name}.onnx')
"
```

### 9.2 ATC 编译

```bash
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.0.0/opp
for model in clip_visual jbu_upsampler; do
    atc --model=models/onnx/${model}.onnx --framework=5 \
        --output=om_310B1/${model} \
        --soc_version=Ascend310B1 --log=info
done
# clip_text 需要指定固定 batch（ONNX 有动态 batch axis）
atc --model=models/onnx/clip_text.onnx --framework=5 \
    --output=om_310B1/clip_text \
    --soc_version=Ascend310B1 --input_shape="text:8,77" --log=info
```

### 9.3 产物验证

| 模型 | 310B1 大小 | 910 大小 | ATC 警告 |
|------|-----------|----------|---------|
| `clip_visual.om` | 163 MB | 162 MB | 无 |
| `clip_text.om` | 123 MB | 123 MB | 无 |
| `jbu_upsampler.om` | 10 MB | 9.8 MB | 无 |

### 9.4 310B1 显存预估

JBU combined workspace 在 910 上为 6.3 GB，310B1 HBM 容量通常为 8 GB。
如实际 workspace 同等大小，需降级为 `num_levels=2`（2 级 JBU）或混合后端。

## 10. Git 提交历史

```
cb916d3 fix: 消除 ArgMax 算子 — 用 ReduceMax+Equal+Where 等价替代
cc66c08 feat: 310B1 ATC 编译成功 — onnx-simplifier 消除 Mod 常量算子
31d0d58 exp: ATC 编译选项对 JBU workspace 影响测试 — 无显著优化
6e38612 chore: HBM 逐组件测量脚本 + 定位 JBU workspace 6.3 GB 瓶颈
a4c330a fix: context 恢复 + benchmark 容错 double-free 退出码
4f176af perf: 显存优化 — text encoder 用后即卸 + close() 补全
530c435 feat: AclHybridSession — Visual OM + JBU PyTorch 混合后端
bf59a87 feat: Im2col InferShape 解决 GE shape 传播, ATC 全链路打通
4ba9fcc docs: 记录 Im2col ONNX parser 插件的进展与限制
23981c1 feat: Im2col ONNX parser 插件 (K=11 硬编码)
a8a5098 perf: adaptive_conv 改为逐位置累加, 消除 ConcatD + ReduceSumD
1b656ea perf: logits 计算从 numpy CPU 移到 NPU torch + context 恢复
627efe2 perf: _OmModel buffer 预分配, 消除逐次 malloc/free
cc9f9e7 chore: 确认 im2col 优化后 OM 推理结果正确的基线
025b0bc feat: per-stage OM 推理 + 三后端综合 benchmark
682a29a docs: 补充 Unfold→Slice 优化结果及分析
6d83b0e perf: 用 slice+stack 替代 Unfold 消除 GatherV2 瓶颈
```
