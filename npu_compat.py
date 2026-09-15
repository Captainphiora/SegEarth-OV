"""Ascend NPU compatibility helpers (Ascend310B / Atlas 200I A2).

310B 与 910 的算子能力差别较大，这里集中处理两件事：

1. Cube 单元不支持 FP32 matmul。aclnnMatmul 在 cubeMathType=KEEP_DTYPE 时
   直接报 AclNN_Parameter_Error(EZ1001, error code 161002)。改成
   ALLOW_FP32_DOWN_PRECISION 后 FP32 输入会自动降精度到 FP16 计算。
2. Cube 只有 ND 格式的 MatMulV2 kernel，FRACTAL_NZ 内部格式会报
   "Cannot find bin of op MatMulV2"，因此关闭 allow_internal_format。
3. 没有 FlashAttention。torch_npu 的 scaled_dot_product_attention 会走
   aclnnFlashAttentionScore（error code 161001 not supported），并且它的
   math 回退路径会产出 fp16xfp16->fp32 的 BatchMatMulV2（无对应 kernel）。
   这里用纯 bmm + softmax 的实现替换 F.scaled_dot_product_attention，
   nn.MultiheadAttention 也会自动走这条路径。

在任何 NPU 计算之前调用 setup_npu() 即可。
"""

import math
import os

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover - 非 NPU 环境
    torch_npu = None

_INITIALIZED = False

# 这些 SoC 的 Cube 只有 FP16 kernel，且不支持 FlashAttention
LEGACY_SOC = ('310B', '310P')


def get_device_name():
    if torch_npu is None or not torch.npu.is_available():
        return ''
    return torch.npu.get_device_name(0)


def is_legacy_soc(device_name=None):
    name = device_name if device_name is not None else get_device_name()
    return any(soc in name for soc in LEGACY_SOC)


def sdpa_math(query, key, value, attn_mask=None, dropout_p=0.0,
              is_causal=False, scale=None, enable_gqa=False):
    """F.scaled_dot_product_attention 的纯 matmul/softmax 实现。

    只使用 310B 支持的算子，且严格保持输入 dtype，避免出现
    fp16 x fp16 -> fp32 的 matmul。
    """
    scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale

    if enable_gqa and key.shape[-3] != query.shape[-3]:
        repeats = query.shape[-3] // key.shape[-3]
        key = key.repeat_interleave(repeats, dim=-3)
        value = value.repeat_interleave(repeats, dim=-3)

    attn = torch.matmul(query, key.transpose(-2, -1)) * scale

    if is_causal:
        causal = torch.ones(query.shape[-2], key.shape[-2], dtype=torch.bool,
                            device=query.device).tril()
        attn = attn.masked_fill(~causal, float('-inf'))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attn_mask, float('-inf'))
        else:
            attn = attn + attn_mask.to(attn.dtype)

    attn = torch.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = torch.dropout(attn, dropout_p, True)

    return torch.matmul(attn, value)


def patch_sdpa():
    """把 torch 的 SDPA 换成 sdpa_math（含 nn.MultiheadAttention 路径）。"""
    torch.nn.functional.scaled_dot_product_attention = sdpa_math


_orig_linear = torch.nn.functional.linear


def linear_matmul(input, weight, bias=None):
    """用 matmul + add 替代 addmm。

    aclnnAddmm 在 310B 上对 fp16 缺少 MatMulV2 kernel
    ("Cannot find bin of op MatMulV2")，拆开写可以正常跑。
    """
    out = torch.matmul(input, weight.transpose(-2, -1))
    if bias is not None:
        out = out + bias
    return out


def patch_linear():
    torch.nn.functional.linear = linear_matmul


_orig_linspace = torch.linspace


def linspace_cpu(*args, **kwargs):
    """aclnnLinspace 在 310B 上未实现，改为 CPU 生成后拷贝到 NPU。"""
    device = kwargs.pop('device', None)
    out = _orig_linspace(*args, **kwargs)
    if device is not None:
        out = out.to(device)
    return out


def patch_linspace():
    torch.linspace = linspace_cpu


_orig_pad = torch.nn.functional.pad


def _reflect_pad_1d(x, left, right, dim):
    if left == 0 and right == 0:
        return x
    n = x.shape[dim]
    idx = (list(range(left, 0, -1)) + list(range(n))
           + list(range(n - 2, n - 2 - right, -1)))
    index = torch.tensor(idx, dtype=torch.long, device=x.device)
    return x.index_select(dim, index)


def pad_reflect_compat(input, pad, mode='constant', value=None):
    """aclnnReflectionPad2d / aclnnFlip 在 310B 上都不可用，改用 index_select。"""
    if mode == 'reflect' and input.is_npu and len(pad) == 4:
        out = _reflect_pad_1d(input, pad[0], pad[1], -1)
        return _reflect_pad_1d(out, pad[2], pad[3], -2)
    return _orig_pad(input, pad, mode=mode, value=value)


def patch_pad():
    torch.nn.functional.pad = pad_reflect_compat


_orig_unfold = torch.nn.functional.unfold


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def unfold_compat(input, kernel_size, dilation=1, padding=0, stride=1):
    """aclnnIm2col 在 310B 上不可用，用 index_select + reshape 实现 unfold。"""
    if not input.is_npu or input.dim() != 4:
        return _orig_unfold(input, kernel_size, dilation, padding, stride)

    kh, kw = _pair(kernel_size)
    dh, dw = _pair(dilation)
    ph, pw = _pair(padding)
    sh, sw = _pair(stride)

    x = input
    if ph or pw:
        x = _orig_pad(x, (pw, pw, ph, ph))

    n, c, h, w = x.shape
    oh = (h - dh * (kh - 1) - 1) // sh + 1
    ow = (w - dw * (kw - 1) - 1) // sw + 1

    idx_h = (torch.arange(oh)[:, None] * sh
             + torch.arange(kh)[None, :] * dh).reshape(-1).to(x.device)
    idx_w = (torch.arange(ow)[:, None] * sw
             + torch.arange(kw)[None, :] * dw).reshape(-1).to(x.device)

    x = x.index_select(2, idx_h).reshape(n, c, oh, kh, w)
    x = x.index_select(4, idx_w).reshape(n, c, oh, kh, ow, kw)
    x = x.permute(0, 1, 3, 5, 2, 4).reshape(n, c * kh * kw, oh * ow)
    return x


def patch_unfold():
    torch.nn.functional.unfold = unfold_compat


_orig_interpolate = torch.nn.functional.interpolate


def _bilinear_upsample_npu(input, size, align_corners):
    """纯 NPU 可分离双线性上采样：先插值高度，再插值宽度。"""
    _, _, h_in, w_in = input.shape
    h_out, w_out = size
    device = input.device
    dtype = input.dtype

    if align_corners:
        h_scale = (h_in - 1) / (h_out - 1) if h_out > 1 else 0.0
        w_scale = (w_in - 1) / (w_out - 1) if w_out > 1 else 0.0
        src_h = torch.arange(h_out, device=device).float() * h_scale
        src_w = torch.arange(w_out, device=device).float() * w_scale
    else:
        h_scale = h_in / h_out
        w_scale = w_in / w_out
        src_h = ((torch.arange(h_out, device=device).float() + 0.5) * h_scale - 0.5).clamp(min=0)
        src_w = ((torch.arange(w_out, device=device).float() + 0.5) * w_scale - 0.5).clamp(min=0)

    h0 = src_h.long().clamp(max=h_in - 1)
    h1 = (h0 + 1).clamp(max=h_in - 1)
    w0 = src_w.long().clamp(max=w_in - 1)
    w1 = (w0 + 1).clamp(max=w_in - 1)

    wh = (src_h - h0.float()).to(dtype).reshape(1, 1, h_out, 1)
    ww = (src_w - w0.float()).to(dtype).reshape(1, 1, 1, w_out)

    h_interp = input.index_select(2, h0) * (1 - wh) + input.index_select(2, h1) * wh
    del input
    result = h_interp.index_select(3, w0) * (1 - ww) + h_interp.index_select(3, w1) * ww
    del h_interp
    return result


def interpolate_compat(input, size=None, scale_factor=None, mode='nearest',
                       align_corners=None, recompute_scale_factor=None,
                       antialias=False):
    """bilinear/bicubic 在 310B 上部分场景不可用，使用纯 NPU 基础算子实现 bilinear。"""
    if input.is_npu and mode in ('bilinear', 'bicubic'):
        try:
            return _orig_interpolate(
                input, size=size, scale_factor=scale_factor, mode=mode,
                align_corners=align_corners,
                recompute_scale_factor=recompute_scale_factor,
                antialias=antialias)
        except Exception:
            if size is None:
                sf = scale_factor if isinstance(scale_factor, (tuple, list)) \
                    else (scale_factor, scale_factor)
                size = (int(input.shape[2] * sf[0]), int(input.shape[3] * sf[1]))
            ac = align_corners if align_corners is not None else False
            return _bilinear_upsample_npu(input, size, ac)
    return _orig_interpolate(
        input, size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
        antialias=antialias)


def patch_interpolate():
    torch.nn.functional.interpolate = interpolate_compat


def setup_npu(verbose=True, use_interpolate_patch=True, use_patch=True):
    """按当前 SoC 型号设置 NPU 运行参数，返回设备名。

    Args:
        verbose: 是否打印初始化信息。
        use_interpolate_patch: 是否启用手动 bilinear interpolate patch。
            设为 False 则使用 PyTorch 原始 interpolate。
        use_patch: 是否启用所有算子 patch。设为 False 则只配置 FP16
            精度参数(cube_math_type/allow_hf32/allow_internal_format)，
            不替换任何算子实现。
    """
    global _INITIALIZED
    if torch_npu is None or not torch.npu.is_available():
        return ''

    device_name = get_device_name()
    if _INITIALIZED:
        return device_name

    if is_legacy_soc(device_name) or os.environ.get('FORCE_310B_COMPAT') == '1':
        torch.npu.matmul.cube_math_type = \
            torch_npu.npu.CubeMathType.ALLOW_FP32_DOWN_PRECISION
        torch.npu.conv.allow_hf32 = False
        torch.npu.config.allow_internal_format = False

        if use_patch:
            patch_sdpa()
            patch_linear()
            patch_linspace()
            patch_pad()
            patch_unfold()
            if use_interpolate_patch:
                patch_interpolate()

        patched_list = ''
        if use_patch:
            patched_list = 'SDPA/linear/linspace/pad/unfold'
            if use_interpolate_patch:
                patched_list += '/interpolate'
        if verbose:
            msg = (f'[npu_compat] {device_name}: cube_math_type='
                   'ALLOW_FP32_DOWN_PRECISION, allow_internal_format=False')
            if patched_list:
                msg += f', {patched_list} patched'
            else:
                msg += ', all patches DISABLED (FP16 precision only)'
            print(msg)

    _INITIALIZED = True
    return device_name
