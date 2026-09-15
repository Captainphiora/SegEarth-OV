# 复现过程问题记录

本文档记录复现 SegEarth-OV 昇腾离线推理部署时遇到的问题及解决方案。
目标是将这些问题逐一消除，使 README.md 中的流程可以一键跑通。

---

## 问题 1: CANN 环境版本不匹配

**现象**

使用 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 初始化环境后，
执行 `torch.Tensor.npu()` 报错：

```
RuntimeError: AclGetCannAttribute(ACL_CANN_ATTR_INF_NAN, &enable), error code is 500000
```

**原因**

本机 `ascend-toolkit/latest` 软链接指向 CANN 8.3.RC1，
而 SegEarth 项目的 Conda 环境中 torch_npu 2.8.0.post4 需要 CANN 9.0.0。
两者 ABI 不兼容。

**解决**

使用 CANN 自带的 `set_env.sh` 初始化环境：

```bash
export CANN_HOME=/usr/local/Ascend/cann-9.0.0
```

所有 step 脚本已改为 source 此文件，不再依赖 `ascend-toolkit/set_env.sh`。

**状态**: 已解决，脚本已修正。

---

## 问题 2: JBU Upsampler ONNX 导出失败

**现象**

Step 1 中 `clip_visual` 和 `clip_text` 导出成功，但 `jbu_upsampler` 导出时报错：

```
RuntimeError: Expected all tensors to be on the same device,
but got tensors is on cpu, different from other tensors on npu:0
```

**原因**

JBU 模块内部使用 `torch.meshgrid` 生成坐标网格。
`torch.onnx.export` 的 legacy TorchScript exporter 在 trace 时
将 `meshgrid` 的输出 tensor 放在 CPU 上，而模型其他 tensor 在 NPU 上，
导致后续 `torch.cat` 操作报设备不一致。

这是 ONNX legacy exporter 的 trace 行为问题，与 torch_npu 无关。
`clip_visual` 和 `clip_text` 不涉及 `meshgrid`，所以导出正常。

**当前解决**

使用之前导出的缓存 ONNX 文件（`models/onnx/jbu_upsampler.onnx`），
拷贝到 `models/onnx/` 继续后续流程。

**彻底解决方向**

- 在导出前 monkey-patch `torch.meshgrid`，强制输出 tensor 留在模型所在设备
- 或使用 PyTorch 2.9+ 默认的 `torch.onnx.export(dynamo=True)` 新导出器
- 或将 JBU 模型搬到 CPU 上导出（纯 CPU trace 不存在 device mismatch）

**状态**: 未彻底解决，当前用缓存 ONNX 绕过。

---

## 问题 3: ATC SOC_VERSION 选择错误

**现象**

使用 `--soc_version=Ascend910B4` 编译报大量算子不支持：

```
Unsupported_Operator(EZ3003): No supported Ops kernel and engine
are found for [/conv1/Conv], optype [Conv2D]
```

Conv2D、LayerNorm、Adds 等基础算子全部报找不到 kernel。

**原因**

`Ascend910B4` 是泛化型号名。CANN 9.0.0 的 OPP 算子库目录以具体硬件型号
`ascend910_93` 命名，ATC 在 `Ascend910B4` 下找不到对应的算子 kernel 目录。

**解决**

使用 `torch.npu.get_device_name()` 查询到的具体芯片型号：

```bash
--soc_version=Ascend910_9382
```

step2 脚本已将默认值改为 `Ascend910_9382`。

**状态**: 已解决，脚本已修正。

---

## 问题 4: ACL 预处理与 PyTorch 预处理不一致导致精度差异

**现象**

PyTorch (eval.py) mIoU=50.55%，ACL mIoU=49.66%，差 0.89 个点，
其中 building 类差 2.43 个点。

**原因**

`eval_acl.py` 的 `preprocess_numpy` 原始实现：
1. 先 normalize（减均值除标准差）
2. 再用 PIL.BILINEAR 逐通道 resize

而 MMSeg pipeline（eval.py 使用）：
1. 先用 cv2.INTER_LINEAR resize 原始 uint8 图像
2. 再 normalize

两个差异叠加：插值库不同（PIL vs cv2）+ resize/normalize 顺序颠倒。
在归一化后的浮点数上做插值与在原始 uint8 上做插值，数值结果不同。

**解决**

将 `preprocess_numpy` 改为与 MMSeg 一致的流程：
cv2.resize(uint8) → float32 → normalize → transpose → fp16。

修正后 mIoU 从 49.66% → 50.56%，与 PyTorch 的 50.55% 完全对齐（差 0.01）。

**状态**: 已解决，`scripts/eval_acl.py` 已修正。

---

## 问题 5: 310B1 交叉编译需要专用 ONNX 导出脚本

**现象**

使用 `scripts/export_onnx.py (基础版)` 导出的 ONNX 在 ATC `--soc_version=Ascend310B1` 编译时：
- `clip_text.om`：编译成功但警告 `ArgMaxV2 does not hit the high-priority operator information library`（性能差）
- `jbu_upsampler.om`：编译超时（10 分钟+未完成），因 ONNX Resize(half_pixel) 和 Gather 算子展开后图极大

**原因**

`scripts/export_onnx.py (基础版)` 是面向 910 的基础版，使用了 310B1 不支持或性能极差的算子：
1. `text.argmax(dim=-1)` → ONNX ArgMax → ATC ArgMaxV2（310B1 无高优 kernel）
2. JBU 中 `F.interpolate(mode='bicubic')` → ONNX Resize(half_pixel)（ATC 不支持此坐标模式）
3. JBU 中 `F.unfold` → ONNX Gather → ATC GatherV2（极慢）

**解决**

使用 `scripts/export_onnx.py`（310B 优化版），做了三个等价替换：
1. ArgMax → `ReduceMax + Equal + Where + ReduceMax`（纯比较算子，全平台支持）
2. bicubic 2x 上采样 → depthwise conv2d 等价实现（预计算权重，ATC 高效）
3. Unfold → K² 次 Slice + Mul + Add 累加（全部是 ATC 高效算子）

替换后 310B1 编译全部通过，无警告，JBU 编译约 8 分钟完成。

step1 脚本已改为调用 `scripts/export_onnx.py`。

**状态**: 已解决，step1 脚本已修正。
