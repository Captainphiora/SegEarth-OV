# SegEarth-OV 昇腾 NPU 离线推理部署指南

> PyTorch → ONNX → OM（ATC 编译）→ ACL 推理 完整流程

## 概述

SegEarth-OV 是基于 CLIP ViT-B/16 的开放词表语义分割模型。部署时将其拆分为三个子网络，
分别导出、编译和推理：

| 子模型 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `clip_visual` | 视觉编码器 | `image [1,3,224,224]` fp16 | `cls_token [1,512]` + `patch_tokens [1,196,512]` |
| `clip_text` | 文本编码器 | `text [8,77]` int64 | `text_features [8,512]` fp16 |
| `jbu_upsampler` | JBU 特征上采样器 | `source [1,512,14,14]` + `guidance [1,3,224,224]` | `upsampled [1,512,224,224]` |

## 环境信息

| 项目 | 版本 |
|------|------|
| 硬件 | Ascend 910_9382 (62GB HBM) |
| CANN | 9.0.0 |
| PyTorch | 2.8.0 |
| torch_npu | 2.8.0.post4 |
| Python | 3.10 |
| Conda 环境 | SegEarth |
| SOC_VERSION | Ascend910_9382 |

## 前置条件

1. 昇腾 NPU 环境就绪，CANN 9.0.0 安装完成
2. Conda 环境 `SegEarth` 已创建，含 `torch_npu`、`open_clip`、`mmseg` 等依赖
3. 模型权重已下载：`simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt`
4. 评估数据集已准备：`data/UDD/UDD/UDD5/val/`

## 目录结构

```
scripts/
├── README.md                  # 本文档（流程说明）
├── ISSUES.md                  # 复现过程中遇到的问题及解决方案
├── step0_eval_pytorch.sh      # Step 0: PyTorch 基线评估 (原始 eval.py)
├── step1_export_onnx.sh       # Step 1: PyTorch → ONNX 导出
├── step2_build_om.sh          # Step 2: ONNX → OM 编译 (ATC)
├── step3_eval_acl.sh          # Step 3: OM 推理评估 (ACL)
├── onnx_output/               # ONNX 模型产出
│   ├── clip_visual.onnx
│   ├── clip_text.onnx
│   └── jbu_upsampler.onnx
├── om_output/                 # OM 模型产出
│   ├── clip_visual.om
│   ├── clip_text.om
│   ├── jbu_upsampler.om
│   └── query_features_*.npy   # 文本特征缓存（首次推理自动生成）
└── logs/                      # 推理日志
```

## Step 0: PyTorch 基线评估

```bash
bash scripts/step0_eval_pytorch.sh
```

使用原始 `eval.py`（MMEngine Runner 全流程）在 NPU 上跑 PyTorch 推理，
作为 OM 推理的精度基线。数据预处理、指标计算完全由 MMSeg pipeline 驱动。

## Step 1: PyTorch → ONNX 导出

```bash
bash scripts/step1_export_onnx.sh
```

调用 `scripts/export_onnx.py`（310B 优化版），在 NPU 上加载 CLIP ViT-B/16（fp16），
通过 `torch.onnx.export`（opset 17）将三个子网络分别导出为 ONNX：

- **视觉编码器**：复现 SegEarth 自定义注意力（qq+kk+vv 三路 softmax），最后一层 ignore_residual
- **文本编码器**：标准 CLIP 文本编码 + L2 归一化，batch 维度设为动态
- **JBU 上采样器**：将 14×14 patch features 通过 4 次 ×2 上采样恢复到 224×224

相比基础版 `scripts/export_onnx.py`，310B 优化版做了三个关键适配：

1. **ArgMax 替代**：用 `ReduceMax + Equal + Where + ReduceMax` 替代 `argmax`（310B1 无高优 ArgMaxV2 kernel）
2. **bicubic 上采样**：用 depthwise conv2d 等价实现（ATC 不支持 ONNX Resize 的 half_pixel 模式）
3. **Unfold 替代**：用 slice 累加替代 ONNX Gather → ATC GatherV2（极慢）

另外 Monkey-patch `LayerNormFp32.forward` 使其在 fp16 下直接计算，
避免导出 fp32 cast 算子（310B 等推理卡不支持）。

产出：`models/onnx/` 下三个 `.onnx` 文件。

## Step 2: ONNX → OM 编译

```bash
bash scripts/step2_build_om.sh
```

使用昇腾 ATC（Ascend Tensor Compiler）将 ONNX 编译为 OM 离线模型。
核心命令示例：

```bash
atc --model=clip_visual.onnx \
    --framework=5 \
    --output=clip_visual \
    --soc_version=Ascend910_9382 \
    --input_shape="image:1,3,224,224" \
    --input_format=NCHW
```

三个模型的编译参数：

| 模型 | input_shape | 说明 |
|------|------------|------|
| clip_visual | `image:1,3,224,224` | 静态 batch=1 |
| clip_text | `text:8,77` | 固定 batch=8，推理时 pad 到 8 |
| jbu_upsampler | `source:1,512,14,14;guidance:1,3,224,224` | 双输入 |

编译耗时约 5 分钟（visual ~1min, text ~30s, jbu ~3.5min）。
产出：`models/om/` 下三个 `.om` 文件。

## Step 3: ACL 推理评估

```bash
bash scripts/step3_eval_acl.sh
```

使用 ACL（Ascend Computing Language）加载 OM 模型进行语义分割推理：

1. **初始化 ACL 环境**：`acl.init()` → `set_device` → `create_context`
2. **文本特征编码**（仅首次）：加载 `clip_text.om`，对类别名 × 80 个模板编码，结果缓存为 `.npy`，后续直接加载
3. **逐图推理**：滑窗裁切(stride=112, crop=224) → `clip_visual.om` → `jbu_upsampler.om` → 与文本特征 einsum → softmax → argmax
4. **精度评估**：计算 mIoU / aAcc / mAcc

可通过脚本中的变量调整行为：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NPU_ID` | 4 | 使用的 NPU 设备编号 |
| `MAX_SAMPLES` | 5 | 评估样本数，0 = 全量 |
| `TEMPLATE` | full | 文本模板，full(80) 或 sub(7) |

评估结果（UDD5 数据集，全量 40 张）：

### PyTorch vs ACL 精度对比

| 指标 | PyTorch (eval.py) | ACL (OM) | 差值 |
|------|-------------------|----------|------|
| aAcc | 74.86% | 74.87% | +0.01 |
| mIoU | 50.55% | 50.56% | +0.01 |
| mAcc | 66.15% | 66.16% | +0.01 |

逐类 IoU 对比：

| 类别 | PyTorch | ACL | 差值 |
|------|---------|-----|------|
| vegetation | 80.53 | 80.52 | -0.01 |
| building | 70.93 | 70.98 | +0.05 |
| road | 53.31 | 53.30 | -0.01 |
| vehicle | 21.55 | 21.55 | 0.00 |
| background | 26.44 | 26.44 | 0.00 |

两个后端精度完全对齐，所有指标差异 ≤ 0.05 个点。

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    PyTorch 模型                          │
│  CLIP ViT-B/16 (Visual) + CLIP Text + JBU Upsampler    │
└──────────────┬──────────────────────────────────────────┘
               │  torch.onnx.export (opset 17, fp16)
               ▼
┌─────────────────────────────────────────────────────────┐
│                    ONNX 模型 (×3)                        │
│  clip_visual.onnx | clip_text.onnx | jbu_upsampler.onnx │
└──────────────┬──────────────────────────────────────────┘
               │  ATC --framework=5 --soc_version=Ascend910_9382
               ▼
┌─────────────────────────────────────────────────────────┐
│                    OM 离线模型 (×3)                       │
│  clip_visual.om  |  clip_text.om  |  jbu_upsampler.om   │
└──────────────┬──────────────────────────────────────────┘
               │  ACL Runtime (acl.mdl.execute)
               ▼
┌─────────────────────────────────────────────────────────┐
│                    ACL 推理部署                           │
│  文本特征缓存 → 滑窗推理 → visual.om → upsampler.om      │
│  → einsum 相似度 → softmax → argmax → 语义分割结果       │
└─────────────────────────────────────────────────────────┘
```

### 设计要点

1. **三模型拆分**：文本编码器只需运行一次并缓存，推理时仅需 visual + upsampler 两个 OM
2. **滑窗推理**：stride=112, crop=224 遍历全图，处理任意分辨率遥感影像
3. **同义词支持**：每个类别可配置多个名称（逗号分隔），在 postprocess 中用 max-pool 合并
4. **fp16 全链路**：从 PyTorch 导出到 OM 推理全程 fp16，匹配推理卡能力
