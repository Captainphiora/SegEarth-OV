# SegEarth-OV 310B1 ACL 纯推理适配记录

> 日期: 2026-09-01 · 分支: `310B` · 设备: Atlas 200I DK A2 (Ascend 310B1, 11577 MB)

## 目标

在 Ascend 310B1 上通过 `python scripts/demo_multi.py --backend acl --size 448` 完成
SegEarth-OV 遥感语义分割推理，使用纯 ACL OM 后端，输入 448×448，输出分割图。

## 遇到的问题与解决过程

### 1. Python 模块冲突

**现象**: `ModuleNotFoundError: No module named 'utils.session'; 'utils' is not a package`

**原因**: 项目根目录同时存在 `utils.py`（早期遗留的 Excel 工具）和 `utils/`（session
模块所在包），Python 优先导入 `utils.py` 文件，遮盖了 `utils/` 包。

**修复**:
- 将 `utils.py` 重命名为 `utils_legacy.py`
- 为 `utils/` 添加 `__init__.py`

### 2. FP32 矩阵运算不支持

**现象**: `aclnnBatchMatMul failed, error code 107004` — "npu arch does not support
FP32 for calculations when cubeMathType is KEEP_DTYPE"

**原因**: 310B1 的 Cube 单元不支持 FP32 矩阵乘法。`_forward_feature` 中的
`torch.einsum("bnd,cd->bnc", ...)` 和 `ct @ self._qf_npu.T` 使用了 float32 张量。

**修复**: 后续在去 torch 依赖时一并解决（改为纯 numpy float32 计算，走 CPU 而非
NPU Cube 单元）。

### 3. NPU 显存 OOM（核心问题）

这是整个适配过程中最关键的问题，经过三轮内存监控逐步定位。

#### 第一轮：全量文本编码爆显存

使用 `npu-smi` 每 2 秒采样，发现文本编码阶段 NPU 显存从 5.9 GB 飙升至 11 GB：

```
时间     NPU 内存      系统可用    阶段
04:02:56  5941 MB      6375 MB    text OM 加载完毕
04:03:22 10952 MB      1784 MB    文本编码中途 ← SWAP 开始
04:03:33 11085 MB      1285 MB    ← Segfault
```

原因: `openai_imagenet_template` 有 80 个提示模板，14 个 query words × 80 模板 /
batch_size 8 = **140 次 OM 推理调用**，每次 `acl.rt.malloc` / `acl.rt.free` 产生的
显存碎片在 ACL runtime 内累积。

尝试: 将模板从 80 个 (`openai_imagenet_template`) 缩减到 7 个
(`sub_imagenet_template`)，调用次数降到 14 次。

#### 第二轮：减模板后仍然 OOM

监控发现即使只有 14 次文本 OM 调用，NPU 仍然从 5 GB 飙到 11 GB。说明问题不仅是
调用次数，而是 **text OM 模型本身的 workspace + torch_npu 运行时 + visual OM +
JBU OM 的总和**超出了 310B1 的 11.5 GB 容量。

尝试: 预计算 query features 缓存到 `.npy` 文件，推理时直接加载，完全跳过 text OM。

#### 第三轮：去 text OM 后仍然 OOM

预计算缓存生效后，text OM 不再加载，但 NPU 仍然逼近 11 GB。分析内存构成：

| 组件                   | 估算显存    |
|------------------------|-------------|
| ACL runtime 基线       | ~2.7 GB     |
| torch_npu 运行时初始化 | **~2.0 GB** |
| clip_visual.om + workspace | ~2.5 GB |
| jbu_upsampler.om + workspace | ~0.5 GB |
| 推理过程临时显存       | ~1-2 GB     |
| **合计**               | **~8.7-9.7 GB** |

关键发现: **torch_npu 的 import 和初始化本身就占约 2 GB 显存**，即使只用了
`F.normalize` 和 `einsum` 这两个简单操作。

最终方案: ACL 后端**完全去除 torch/torch_npu 依赖**，全部改用纯 numpy 实现。

## 最终修改清单

### `utils/session.py` — AclSession 重构

| 修改点 | 原实现 | 新实现 |
|--------|--------|--------|
| 文本特征 | 每次推理加载 text OM 计算 | 预计算缓存到 `.npy`，首次运行后自动复用 |
| `_qf_npu` | `torch.Tensor` on NPU | `numpy.ndarray` (float32) on CPU |
| `F.normalize` | torch on NPU | `np.linalg.norm` on CPU |
| `torch.einsum` | torch on NPU (FP16 matmul) | `np.einsum` on CPU |
| `F.interpolate` | torch bilinear on NPU | `PIL.Image.resize` on CPU |
| `close()` | 无保护，double free | 添加 `_closed` 标志防重复释放 |

### `scripts/demo_multi.py` — ACL 分支去 torchvision

| 修改点 | 原实现 | 新实现 |
|--------|--------|--------|
| 图像预处理 | `torchvision.transforms` (触发 torch_npu import) | 纯 PIL + numpy |

### 新增文件

- `scripts/precompute_query_features.py` — 独立预计算脚本
- `utils/__init__.py` — 包标识
- `AGENTS.md` — 项目环境规则（清显存 / CANN / conda）

### 重命名

- `utils.py` → `utils_legacy.py`（避免包名冲突）

## 内存对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| NPU 峰值 | 11,085 MB (Segfault) | ~6,950 MB |
| 显存余量 | 0 (OOM) | ~4,600 MB |
| SWAP 使用 | 190+ MB | 0 MB |
| 退出状态 | Segfault / Aborted | 正常退出 (code 0) |

## 使用方法

```bash
# 1. 环境准备
sudo bash scripts/cleanup_memory.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
eval "$(conda shell.bash hook)" && conda activate SegEarth

# 2. 首次运行需预计算文本特征（仅一次，后续自动缓存）
python scripts/precompute_query_features.py

# 3. 运行推理
python scripts/demo_multi.py --backend acl --size 448
# 输出: results/acl/seg_pred.png
```

如果更换类名列表，需删除 `models/om/query_features_*.npy` 缓存后重新
运行预计算脚本。

## 经验总结

1. **310B1 显存是硬约束**: 11.5 GB 总量中基线占 2.7 GB，可用仅 ~8.8 GB。任何多
   余的框架初始化都可能成为压死骆驼的最后一根稻草。
2. **torch_npu 的隐性成本**: import torch_npu 即占约 2 GB 显存，在内存紧张的嵌入
   式设备上应尽量避免。ACL 后端能用 numpy 完成的操作就不要引入 torch。
3. **预计算是最有效的省显存手段**: query features 只取决于类名和模板，与输入图片
   无关。缓存到 14 KB 的 `.npy` 文件即可省掉 text OM 的 2+ GB 显存。
4. **内存监控是定位 OOM 的关键**: 每 2 秒采样 `npu-smi` + `free -m` 能精确定位
   显存在哪个阶段飙升，比猜测有效得多。
