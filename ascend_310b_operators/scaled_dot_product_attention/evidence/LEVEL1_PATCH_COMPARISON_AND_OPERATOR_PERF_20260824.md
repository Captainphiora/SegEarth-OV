# Level-1 Patch 对比与六算子性能 - 2026-08-24

## 测试边界

- 设备：Atlas 200I DK A2，Ascend 310B1，physical 0 / logical `npu:0`
- 环境：`SegEarth`，CANN 9.0.0，torch_npu 2.8.0.post4
- Demo：`num_levels=1`
- 输入、模型权重、类别和 GPU golden 均保持一致
- 完整 demo 使用标准 6 GiB 内存、Swap 0 守护
- 六算子均在所有兼容 patch 关闭时测试
- 算子计时：3 次 warmup，10 次 NPU 同步采样；表中 p95 按现有测试脚本字段记录

## Level-1 Demo 对比

| 配置 | 结果 | 墙钟时间 | 最大子进程 RSS | Cgroup 内存峰值 | NPU 前/后 | Swap |
|---|---|---:|---:|---:|---:|---:|
| `USE_PATCH=True` | PASS | 92.135 s | 2563892 KiB | 5550.0 MiB | 4998 / 4904 MiB | 0 / 0 KiB |
| `USE_PATCH=False` | PASS | 97.110 s | 2560496 KiB | 5687.7 MiB | 4865 / 4904 MiB | 0 / 0 KiB |

本次单次观测中，patch-off 比 patch-on 慢 4.975 秒，即相对 patch-on
增加约 5.40%。两次输出 SHA256 均为
`bbb3cef850196c02a6b5b77eca7d6b56dd0d00a85a8a598377f53d61cf2a7f05`，
RGBA 像素完全一致，MAE/RMSE/最大差均为 0。由于完整 demo 每种配置只测
一次，时长差可能包含编译缓存、设备基线和系统抖动，不能单独作为稳定性能
回归结论。

patch-off 输出保存为 `seg_pred_num_levels_1_patch_off.png`。相对
`seg_pred_gpu_golden.png`：完全一致像素比例 0.9415870831，MAE
3.7811673693，RMSE 22.7405969278，PSNR 20.9947663990 dB，最大通道差
230。

## 六算子独立性能

| 算子 | 代表输入/属性 | 正确性案例 | 最大误差 | Median | P95 | NPU 前/后 | Swap |
|---|---|---:|---:|---:|---:|---:|---:|
| Linspace | 257 个 FP16 元素 | 5 | 0.0009765625 | 616.992 us | 4541.451 us | 4918 / 5029 MiB | 0 / 0 KiB |
| Linear | `[64,256] @ [256,256]^T + [256]` FP16 | 3 | 0.0078125 | 358.407 us | 768.398 us | 4958 / 5076 MiB | 0 / 0 KiB |
| ReflectionPad2d | `[1,8,31,29]`, pad `(3,2,4,1)` FP16 | 3 | 0 | 4728.441 us | 8090.358 us | 4965 / 5071 MiB | 0 / 0 KiB |
| Im2col | `[1,8,32,32]`, k3/pad1/stride1 FP16 | 3 | 0 | 16121.559 us | 24025.927 us | 4982 / 5095 MiB | 0 / 0 KiB |
| UpsampleBilinear2d | `[1,8,32,32] -> [1,8,64,64]` FP16 | 4 | 0.0010986328 | 12644.403 us | 38552.492 us | 4984 / 5104 MiB | 0 / 0 KiB |
| SDPA | Q/K/V `[1,4,64,64]` FP16 | 4 | 0.0009765625 | 502.387 us | 782.211 us | 4996 / 5091 MiB | 0 / 0 KiB |

六个候选均通过当前契约正确性案例、零 Swap 门和 NPU 显存释放门。按代表
形状的 median，当前最重的是 Im2col，其次是 UpsampleBilinear2d 和
ReflectionPad2d；Linear 与 SDPA 的中位耗时低于 1 ms。Bilinear 和
Linspace 的 p95 波动较大，后续性能优化应优先复测并分析这两个尾延迟，且
不能用单次完整 demo 时间代替算子级重复测量。

## Patch-Off 接入结论

全关过程中补齐了 `open_clip/transformer.py` 的 NPU MHA/Linear 显式调用，
以及 `segearth_segmentor.py` 的五处 bilinear 显式调用。最终
`demo.py` 保持 `USE_PATCH=False`、`USE_INTERPOLATE_PATCH=False` 和
`num_levels=1`。没有使用全局 PyTorch monkey patch、异常回退或 CPU 候选路径。

## 六算子源码地址与使用方法

以下路径均位于设备目录 `/home/chenxinji/SegEarth-OV-NPU`。六个实现都是
显式调用的 NPU 张量组合，不依赖全局 monkey patch；不支持的设备、dtype、
rank 或属性会直接抛出异常，不会回退到 CPU、PyTorch 等价整算子或其他后端。

### 公共初始化

从项目根目录启动 `SegEarth` 环境后，在模型初始化阶段关闭兼容 patch：

```python
import torch
import torch_npu  # noqa: F401

from npu_compat import setup_npu

setup_npu(use_patch=False, use_interpolate_patch=False)
device = torch.device("npu:0")
torch.npu.set_device(device)
```

实际执行应继续通过 `/home/chenxinji/310B_constraints/run_310b.sh`，并在进程
结束后按 310B 约束清理 NPU 显存。

### 1. Scaled Dot-Product Attention

- 源码：`/home/chenxinji/SegEarth-OV-NPU/ascend_310b_operators/scaled_dot_product_attention/source/__init__.py`
- 导入：`from ascend_310b_operators.scaled_dot_product_attention.source import scaled_dot_product_attention_310b`

```python
out = scaled_dot_product_attention_310b(
    query,
    key,
    value,
    attn_mask=mask,
    dropout_p=0.0,
    is_causal=False,
)
```

`query`、`key`、`value` 必须是同 dtype 的四维 NPU FP16 张量；支持布尔或
加性 mask、`scale` 和 GQA。当前推理实现要求 `dropout_p=0.0`，且
`attn_mask` 与 `is_causal=True` 不能同时使用。

### 2. Linear

- 源码：`/home/chenxinji/SegEarth-OV-NPU/ascend_310b_operators/linear_addmm/source/__init__.py`
- 导入：`from ascend_310b_operators.linear_addmm.source import linear_310b`

```python
y = linear_310b(x, weight, bias)
```

计算语义为 `x @ weight.T + bias`。`x`、`weight` 和可选 `bias` 均须位于
NPU 且为 FP16；`x` 形状为 `[..., in_features]`，`weight` 为
`[out_features, in_features]`，`bias` 为 `[out_features]`。

### 3. Linspace

- 源码：`/home/chenxinji/SegEarth-OV-NPU/ascend_310b_operators/linspace/source/__init__.py`
- 导入：`from ascend_310b_operators.linspace.source import linspace_310b`

```python
positions = linspace_310b(
    0.0,
    1.0,
    257,
    dtype=torch.float16,
    device=device,
)
```

`steps` 必须是非负整数，`device` 必须是 NPU，输出 dtype 支持 FP16 和
FP32。数值直接在 NPU 上生成，不先生成 CPU 张量再复制。

### 4. ReflectionPad2d

- 源码：`/home/chenxinji/SegEarth-OV-NPU/ascend_310b_operators/reflection_pad2d/source/__init__.py`
- 导入：`from ascend_310b_operators.reflection_pad2d.source import reflection_pad2d_310b`

```python
padded = reflection_pad2d_310b(x, (3, 2, 4, 1))
```

输入必须是 NPU 上的四维 NCHW FP16/FP32 张量；`pad` 顺序是
`(left, right, top, bottom)`。四个值必须是非负整数，左右 padding 分别
小于输入宽度，上下 padding 分别小于输入高度。

### 5. Im2col

- 源码：`/home/chenxinji/SegEarth-OV-NPU/ascend_310b_operators/im2col/source/__init__.py`
- 导入：`from ascend_310b_operators.im2col.source import im2col_310b`

```python
columns = im2col_310b(
    x,
    kernel_size=(3, 3),
    dilation=1,
    padding=1,
    stride=1,
)
```

输入必须是 NPU 上的四维 NCHW FP16/FP32 张量。卷积属性可传整数或二元
组；返回 PyTorch Unfold 排列的 `[N, C * kernel_h * kernel_w, L]` 张量。

### 6. UpsampleBilinear2d

- 源码：`/home/chenxinji/SegEarth-OV-NPU/ascend_310b_operators/upsample_bilinear2d/source/__init__.py`
- 导入：`from ascend_310b_operators.upsample_bilinear2d.source import upsample_bilinear2d_310b`

```python
resized = upsample_bilinear2d_310b(
    x,
    size=(64, 64),
    align_corners=False,
    antialias=False,
)
```

输入必须是 NPU 上的四维 NCHW FP16/FP32 张量。`size` 与
`scale_factor` 必须且只能提供一个，两者均可传标量或二元组；支持
`align_corners` 和 `recompute_scale_factor`，当前不支持
`antialias=True`。

### 项目内显式接入位置

- SDPA 和 Linear：`open_clip/transformer.py`；Linear 还用于
  `open_clip/modified_resnet.py`。
- Linspace、ReflectionPad2d、Im2col 和 UpsampleBilinear2d：
  `simfeatup_dev/upsamplers.py`。
- UpsampleBilinear2d 的分割输出路径：`segearth_segmentor.py`。

## 第二轮算子优化进展 - 2026-08-24

本轮新增了同进程交替 A/B 性能门：冻结原源码，在一个 guarded 进程内对
baseline/candidate 交替计时；完整正确性通过且两段配对中位数都更快才允许写入
正式源码。所有命令均增加显式超时，Swap 始终为 0。

| 算子 | 结果 | Baseline Median | Candidate Median | 配对加速 | 正式源码 |
|---|---|---:|---:|---:|---|
| Im2col | 接受 | 16047.7 us | 5275.9 us | 3.04x | `as_strided` 窗口视图 |
| UpsampleBilinear2d | 接受 | 16012.5 us | 10134.2 us | 1.58x | 有界坐标/权重元数据缓存 |
| ReflectionPad2d | 接受 | 11527.1 us | 3112.6 us | 3.70x | 有界扁平反射索引 + 单次 gather |
| Linspace | 拒绝 | 3979.2 us | 5705.7 us | 0.70x | 未改 |
| Linear | 拒绝 | 3946.7 us | 5732.2 us | 0.69x | 未改 |
| SDPA | 拒绝 | 4271.2 us | 4260.9 us | 1.002x | 未改；两段为 0.75x/3.27x |

接受项的精度证据：Im2col 三例逐元素精确，真实
`adaptive_conv_py_simple` 最大误差 0.00390625；Bilinear 四例与
`simfeatup.Bilinear.forward` 通过，最大误差 0.001953125；ReflectionPad2d
在 FP16/FP32 六例及真实接入中逐元素精确。

### 当前阻塞

Linear 最后一个拒绝候选结束后，guard 发现 NPU 显存未从 4878 MiB 返回，
稳定在约 6.2 GiB；整机重启后恢复至 2555 MiB。重启后的第一个 SDPA 探针
再次在无残留 Python 进程和 cgroup 的情况下留下约 3450 MiB 系统级 NPU
驻留，超过 128 MiB 清理容差。`npu-smi reset` 和 in-band reset 均不受该
310B1 支持。因此在首次 CANN/NPU 初始化的持久显存生命周期被明确处理前，
不得继续 NPU 命令，也不得把三项未接受候选报告为加速完成。

## 最终验证 - 2026-08-24

guard 已增加每次开机一次的受控 CANN runtime primer，并将本机 CANN 9.0 的
TE/TBE/knowledge-bank 编译并发统一限制为 2。前者在候选运行前建立稳定 NPU
基线，后者把完整 demo 的 6 GiB cgroup 峰值从 OOM 降至 4724 MiB；没有提高
内存上限或启用 Swap。

当前六份源码 SHA256 保存在
`ascend_310b_operators/scaled_dot_product_attention/evidence/final_source_sha256_20260824.txt`。
六个正式算子测试全部通过：

| 算子 | 案例数 | 最大误差 | 最终独立 Median | 清理 |
|---|---:|---:|---:|---|
| SDPA | 4 | 0.0009765625 | 1302.109 us | PASS |
| Linear | 3 | 0.0078125 | 4470.889 us | PASS |
| Linspace | 6 | 0 | 4008.981 us | PASS |
| ReflectionPad2d | 3 | 0 | 7356.554 us | PASS |
| Im2col | 3 | 0 | 10077.801 us | PASS |
| UpsampleBilinear2d | 4 | 0.00244140625 | 8192.015 us | PASS |

绝对时延受设备抖动影响，只作为最终正确性运行记录；加速结论仍以同进程配对
A/B 为准。按六个代表输入的配对中位数求和，baseline 为 55784.4 us，当前
源码为 32446.3 us，组合代表工作量加速约 1.719x。该组合包含三个明确加速
热点、Linear/SDPA 保持原实现，以及为修复扩展精度而变慢的 Linspace。

五个独立真实调用链全部通过：`open_clip.AttentionPool2d`、
`SimpleImplicitFeaturizer.forward`、`simfeatup._reflection_pad2d`、
`adaptive_conv_py_simple`、`simfeatup.Bilinear.forward`。Linear 由完整 demo
覆盖。最终 patch-off level-1 demo 在 84.573 秒完成，输出 `seg_pred.png` 的
SHA256 为
`bbb3cef850196c02a6b5b77eca7d6b56dd0d00a85a8a598377f53d61cf2a7f05`，
与此前通过的 level-1 patch-off 输出逐字节一致；cgroup 峰值 4724 MiB，
Swap 0。demo 退出后系统 NPU 驻留比起始值高 301 MiB，超过 128 MiB 清理
容差，因此最终设备重启清理仍是完成门，不能省略。

## Level-1 预热后推理 - 2026-08-24

在当前自研算子、`USE_PATCH=False`、`USE_INTERPOLATE_PATCH=False`、
`num_levels=1` 下，模型只构建一次，先执行一次完整 `predict` 预热并同步，
随后测量五次完整 `predict`：

| 项目 | 结果 |
|---|---:|
| 模型构建 | 19.788500 s |
| 完整预热 | 25.543623 s |
| 测量 1 | 25.265202 s |
| 测量 2 | 25.253096 s |
| 测量 3 | 25.254709 s |
| 测量 4 | 25.260631 s |
| 测量 5 | 25.258889 s |
| Median | 25.258889 s |
| P95（5 次中的最大值） | 25.265202 s |

五次输出均与预热输出逐元素一致，输出张量 SHA256 为
`6a645278ddfc4f16d1cd832ade3d578d75389a6e05886821291c86648050e8ba`，
shape 为 `[1,448,448]`。预热相对热态中位数只多 0.284734 秒，即热态约
1.011x；本机此前已运行过同一图，编译磁盘/系统缓存不是完全冷态。

完整冷进程 demo 的 84.573 秒还包含 Python 启动、导入、模型构建、首次推理
和 PNG 保存，约为热态单次推理的 3.348x，但两者计时范围不同，不能解释成
纯算子加速。热态运行 cgroup 峰值 4308 MiB，NPU 6719 -> 6666 MiB；任务
cgroup 的 memory+swap 上限与 memory 上限相等，任务不能使用 Swap，但运行
结束时全局 `/proc/swaps` 从 0 变为 256 KiB，且没有存活进程报告 `VmSwap`。
严格全局零 Swap 门因此失败，不能将该轮标记为完全清理通过。

## Level-1 Patch-On 预热对比 - 2026-08-24

重启恢复全局 Swap 0 后，以完全相同的常驻模型脚本测试
`use_patch=True`、`use_interpolate_patch=True`、`num_levels=1`。日志确认
`SDPA/linear/linspace/pad/unfold/interpolate` 全局 patch 实际启用。执行一次完整
预热后测量五次完整 `predict`：

| 项目 | Patch-Off | Patch-On | Patch-On 相对变化 |
|---|---:|---:|---:|
| 模型构建 | 19.788500 s | 20.516200 s | +0.727700 s / +3.68% |
| 完整预热 | 25.543623 s | 26.234976 s | +0.691353 s / +2.71% |
| 热态 Median | 25.258889 s | 25.543709 s | +0.284819 s / +1.13% |
| 热态最大值 | 25.265202 s | 25.546827 s | +0.281625 s |
| Cgroup 内存峰值 | 4308 MiB | 5735 MiB | +1427 MiB / +33.12% |

Patch-On 五次热态样本为 `25.544544`、`25.540178`、`25.543709`、
`25.541123`、`25.546827` 秒。Patch-Off 热态约为 Patch-On 的 1.0113x，
即当前显式自研算子路径在这组稳定 level-1 热态样本中快约 1.13%。两种配置
的输出张量 SHA256 均为
`6a645278ddfc4f16d1cd832ade3d578d75389a6e05886821291c86648050e8ba`，
五次内部重复也逐元素一致。

Patch-On 运行的 Swap 保持 0，cgroup 没有 OOM；但退出后 NPU 显存从 primer
后的 2261 MiB 变为 2812 MiB，增加 551 MiB，超过 128 MiB 清理容差。因此
触发了新协议允许的 `cleanup_npu.sh` 兜底。

项目清理脚本初版会因命令行包含正则文本而错误匹配调用它的 shell，已改为
按 `/proc` 的 `comm + args` 结构化选择真实 Python/torchrun/forkserver 目标，
并增加只读 `--check` 模式。guard 现在先调用 `--check`；存在任何匹配进程就
拒绝兜底，不执行 kill。无匹配目标时，脚本完成页缓存清理后，NPU 从 4087
MiB 降至 2082 MiB，最终复核为 2102 MiB，Swap 0、无残留进程/cgroup。
按用户修订后的协议，该次 patch-on 运行清理完成，不要求精确返回 2261 MiB。
