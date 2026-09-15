"""导出 SegEarth CLIP ViT-B/16 全部组件为 ONNX，用于 310B OM 离线推理。

用法 (在 910 上执行):
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
    python scripts/export_onnx.py

输出:
    models/onnx/clip_visual.onnx   - visual encoder
        input:  image [1, 3, 224, 224] fp16
        output: cls_token [1, 512] fp16, patch_tokens [1, 196, 512] fp16
    models/onnx/clip_text.onnx     - text encoder
        input:  text [N, 77] int64
        output: text_features [N, 512] fp16
    models/onnx/jbu_upsampler.onnx - JBU feature upsampler
        input:  source [1, 512, 14, 14] fp16, guidance [1, 3, 224, 224] fp16
        output: upsampled [1, 512, 224, 224] fp16
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

# 310B 仅支持 FP16，patch LayerNorm 避免 cast 到 float32
import open_clip.transformer as _tf

def _ln_forward_fp16(self, x: torch.Tensor):
    return F.layer_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.bias.to(x.dtype), self.eps)

_tf.LayerNormFp32.forward = _ln_forward_fp16
_tf.LayerNorm.forward = _ln_forward_fp16

from open_clip import create_model, tokenizer
from simfeatup_dev.upsamplers import get_upsampler, JBULearnedRange, adaptive_conv_py_simple

# Patch JBULearnedRange 以适配 ONNX 导出
import simfeatup_dev.upsamplers as _up

def _adaptive_conv_patched(input, filters):
    """Unfold + weighted sum, 用逐位置 slice 累加替代 Unfold。
    避免 ONNX Gather (ATC 映射为 GatherV2, 极慢),
    也避免 stack+sum 产生的 ConcatD + ReduceSumD (占 JBU 48%)。
    改为 K² 次 Slice + Mul + Add, 全部是 ATC 高效算子。
    """
    b, c, h1, w1 = input.shape
    b, h2, w2, f1, f2 = filters.shape
    assert f1 == f2
    kernel_size = int(f1)
    t_filters = filters.reshape(b, h2, w2, f1 * f2).permute(0, 3, 1, 2)
    result = input[:, :, 0:h2, 0:w2] * t_filters[:, 0:1, :, :]
    for idx in range(1, kernel_size * kernel_size):
        i = idx // kernel_size
        j = idx % kernel_size
        result = result + input[:, :, i:i + h2, j:j + w2] * t_filters[:, idx:idx + 1, :, :]
    return result

_up.adaptive_conv_py_simple = _adaptive_conv_patched

def _make_bicubic_2x_kernels(C, dtype, device, a=-0.75):
    """预计算 bicubic 2x 上采样的 4 组 depthwise conv 权重 (常量)。"""
    def cubic(t):
        t = abs(t)
        if t <= 1:
            return (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1
        elif t < 2:
            return a * t ** 3 - 5 * a * t ** 2 + 8 * a * t - 4 * a
        return 0.0

    we = torch.tensor([cubic(1.75), cubic(0.75), cubic(0.25), cubic(1.25)],
                       dtype=dtype, device=device)
    wo = torch.tensor([cubic(1.25), cubic(0.25), cubic(0.75), cubic(1.75)],
                       dtype=dtype, device=device)
    k_ee = (we.view(4, 1) @ we.view(1, 4)).unsqueeze(0).unsqueeze(0).expand(C, 1, 4, 4).contiguous()
    k_eo = (we.view(4, 1) @ wo.view(1, 4)).unsqueeze(0).unsqueeze(0).expand(C, 1, 4, 4).contiguous()
    k_oe = (wo.view(4, 1) @ we.view(1, 4)).unsqueeze(0).unsqueeze(0).expand(C, 1, 4, 4).contiguous()
    k_oo = (wo.view(4, 1) @ wo.view(1, 4)).unsqueeze(0).unsqueeze(0).expand(C, 1, 4, 4).contiguous()
    return k_ee, k_eo, k_oe, k_oo


_BICUBIC_KERNELS = {}

def _bicubic_2x_conv(x):
    """用 depthwise conv2d 实现 bicubic 2x 上采样 (half_pixel 对齐)。
    与 F.interpolate(mode='bicubic', align_corners=False) 数值等价,
    但仅使用 Pad + Conv2D 算子, ATC 完全支持。
    权重在首次调用时预计算并缓存, trace 时为固定常量。
    """
    B, C, H, W = x.shape
    key = (C, x.dtype, x.device)
    if key not in _BICUBIC_KERNELS:
        _BICUBIC_KERNELS[key] = _make_bicubic_2x_kernels(C, x.dtype, x.device)
    k_ee, k_eo, k_oe, k_oo = _BICUBIC_KERNELS[key]

    xp = F.pad(x, [2, 2, 2, 2], mode='replicate')

    ee = F.conv2d(F.pad(x, [2, 1, 2, 1], mode='replicate'), k_ee, groups=C)
    eo = F.conv2d(F.pad(x, [1, 2, 2, 1], mode='replicate'), k_eo, groups=C)
    oe = F.conv2d(F.pad(x, [2, 1, 1, 2], mode='replicate'), k_oe, groups=C)
    oo = F.conv2d(F.pad(x, [1, 2, 1, 2], mode='replicate'), k_oo, groups=C)

    even_row = torch.stack([ee, eo], dim=-1).reshape(B, C, H, 2 * W)
    odd_row = torch.stack([oe, oo], dim=-1).reshape(B, C, H, 2 * W)
    return torch.stack([even_row, odd_row], dim=-2).reshape(B, C, 2 * H, 2 * W)


def _jbu_forward_export(self, source, guidance):
    """单级 JBU forward, 适配 ONNX 导出。
    修正: 与原始 PyTorch 实现对齐 (bicubic, reflect, einsum)。
    bicubic 2x 上采样用 conv2d 等价实现, 避免 ONNX Resize 算子 (ATC 不支持 half_pixel)。
    """
    GB, GC, GH, GW = guidance.shape
    SB, SC, SH, SQ = source.shape
    K = self.diameter
    R = self.radius

    # spatial kernel
    dist_range = torch.linspace(-1, 1, K, device=source.device)
    x, y = torch.meshgrid(dist_range, dist_range)
    patch = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
    spatial_kernel = torch.exp(- patch.square().sum(0) / (2 * self.sigma_spatial.float() ** 2)) \
        .reshape(1, K * K, 1, 1).to(source.dtype)

    # range kernel — 使用 reflect padding (与原始一致)
    # 用 slice 替代 Unfold, 避免 ONNX Gather → ATC GatherV2
    proj_x = self.range_proj(guidance)
    proj_x_padded = F.pad(proj_x, pad=[R] * 4, mode='reflect')
    queries = torch.stack(
        [proj_x_padded[:, :, i:i + GH, j:j + GW]
         for i in range(K) for j in range(K)],
        dim=2
    ).permute(0, 1, 3, 4, 2)
    pos_temp = self.range_temp.exp().clamp_min(1e-4).clamp_max(1e4)
    score = (queries * proj_x.unsqueeze(-1)).sum(dim=1).permute(0, 3, 1, 2)
    range_kernel = F.softmax(pos_temp * score, dim=1)

    combined_kernel = range_kernel * spatial_kernel
    combined_kernel = combined_kernel / combined_kernel.sum(1, keepdim=True).clamp(1e-7)
    combined_kernel = combined_kernel + .1 * self.fixup_proj(torch.cat([combined_kernel, guidance], dim=1))
    combined_kernel = combined_kernel.permute(0, 2, 3, 1).reshape(GB, GH, GW, K, K)

    # bicubic 2x 上采样 (用 conv2d 等价实现, 避免 Resize half_pixel)
    # kernel buffers 由 JBUUpsamplerWrapper.__init__ 注入到 self 上
    B, C = source.shape[:2]
    ee = F.conv2d(F.pad(source, [2, 1, 2, 1], mode='replicate'), self._bk_ee, groups=C)
    eo = F.conv2d(F.pad(source, [1, 2, 2, 1], mode='replicate'), self._bk_eo, groups=C)
    oe = F.conv2d(F.pad(source, [2, 1, 1, 2], mode='replicate'), self._bk_oe, groups=C)
    oo = F.conv2d(F.pad(source, [1, 2, 1, 2], mode='replicate'), self._bk_oo, groups=C)
    even_row = torch.stack([ee, eo], dim=-1).reshape(B, C, SH, 2 * SQ)
    odd_row = torch.stack([oe, oo], dim=-1).reshape(B, C, SH, 2 * SQ)
    hr_source = torch.stack([even_row, odd_row], dim=-2).reshape(B, C, 2 * SH, 2 * SQ)

    hr_source_padded = F.pad(hr_source, pad=[R] * 4, mode='reflect')
    combined_kernel = combined_kernel.to(hr_source_padded.dtype)
    result = _adaptive_conv_patched(hr_source_padded, combined_kernel)
    return result

JBULearnedRange.forward = _jbu_forward_export


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "models", "onnx")


# ============ Visual Encoder ============

class CLIPVisualEncoder(nn.Module):
    """SegEarth visual encoder: 输出 cls_token + patch_tokens。

    完全复现 segearth_segmentor.py 的 forward_feature 中视觉部分:
    - 走全部 transformer blocks
    - 最后一层用 SegEarth custom_attn (qq+kk+vv)
    - ignore_residual=True
    - 输出 cls_token (用于 cls_token_lambda 加权) 和 patch_tokens
    """
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, image):
        v = self.visual
        x = v.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        cls_emb = v.class_embedding.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls_emb, x], dim=1)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for blk in v.transformer.resblocks[:-1]:
            x = blk(x)

        last_blk = v.transformer.resblocks[-1]
        output = self._custom_attn_segearth(last_blk.attn, last_blk.ln_1(x))
        x = last_blk(x)
        _ = x

        output = output.permute(1, 0, 2)  # LND -> NLD
        output = v.ln_post(output)

        pooled, tokens = output[:, 0], output[:, 1:]
        pooled = pooled @ v.proj
        tokens = tokens @ v.proj

        cls_token = pooled / pooled.norm(dim=-1, keepdim=True)
        return cls_token, tokens

    def _custom_attn_segearth(self, attn_layer, x):
        num_heads = attn_layer.num_heads
        num_tokens, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        qkv = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        qq_attn = torch.bmm(q, q.transpose(1, 2)) * scale
        kk_attn = torch.bmm(k, k.transpose(1, 2)) * scale
        vv_attn = torch.bmm(v, v.transpose(1, 2)) * scale
        attn_weights = (F.softmax(qq_attn, dim=-1)
                        + F.softmax(kk_attn, dim=-1)
                        + F.softmax(vv_attn, dim=-1))

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        attn_output = attn_layer.out_proj(attn_output)
        return attn_output


# ============ Text Encoder ============

class CLIPTextEncoder(nn.Module):
    """CLIP text encoder: 输出 L2-normalized text features。

    为避免 ONNX ArgMax 算子 (310B1 无高优实现), 用
    ReduceMax + Equal + Where + ReduceMax 等价替代 argmax。
    """
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.register_buffer('attn_mask', clip_model.attn_mask)

    @staticmethod
    def _argmax_no_argmax(x):
        """用 ReduceMax+Equal+Where+ReduceMax 替代 argmax, 避免 ArgMax 算子。"""
        max_val = x.max(dim=-1, keepdim=True).values
        mask = (x == max_val)
        indices = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)
        masked = torch.where(mask, indices, torch.tensor(-1, device=x.device, dtype=x.dtype))
        return masked.max(dim=-1).values.long()

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        attn_mask = self.attn_mask.to(cast_dtype) if self.attn_mask is not None else None
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        eot_indices = self._argmax_no_argmax(text)
        x = x[torch.arange(x.shape[0]), eot_indices]
        if self.text_projection is not None:
            x = x @ self.text_projection
        x = x / x.norm(dim=-1, keepdim=True)
        return x


# ============ JBU Upsampler ============

def _inject_bicubic_buffers(jbu_module, C=512, dtype=torch.float16, device='npu'):
    """向 JBULearnedRange 模块注入 bicubic 2x 上采样的 depthwise conv 权重。

    _jbu_forward_export 依赖 self._bk_ee/eo/oe/oo 四个 buffer,
    此函数确保无论是 combined 还是 per-stage 导出, 模块都具备这些属性。
    """
    k_ee, k_eo, k_oe, k_oo = _make_bicubic_2x_kernels(C, dtype, device)
    jbu_module._bk_ee = k_ee
    jbu_module._bk_eo = k_eo
    jbu_module._bk_oe = k_oe
    jbu_module._bk_oo = k_oo
    return k_ee, k_eo, k_oe, k_oo


class JBUUpsamplerWrapper(nn.Module):
    """JBUOne upsampler 封装，固定输入尺寸便于导出。

    展开 4 次 upsample 调用，用固定尺寸替代 adaptive_avg_pool2d。
    bicubic 2x 上采样用 depthwise conv2d 等价实现 (权重注册为 buffer)。
    input:  source [1, 512, 14, 14], guidance [1, 3, 224, 224]
    output: upsampled [1, 512, 224, 224]
    """
    def __init__(self, upsampler):
        super().__init__()
        self.up = upsampler.up
        self.fixup_proj = upsampler.fixup_proj

        C = 512
        k_ee, k_eo, k_oe, k_oo = _inject_bicubic_buffers(self.up, C)
        self.register_buffer('bk_ee', k_ee)
        self.register_buffer('bk_eo', k_eo)
        self.register_buffer('bk_oe', k_oe)
        self.register_buffer('bk_oo', k_oo)

    def _bicubic_2x(self, x):
        B, C, H, W = x.shape
        ee = F.conv2d(F.pad(x, [2, 1, 2, 1], mode='replicate'), self.bk_ee, groups=C)
        eo = F.conv2d(F.pad(x, [1, 2, 2, 1], mode='replicate'), self.bk_eo, groups=C)
        oe = F.conv2d(F.pad(x, [2, 1, 1, 2], mode='replicate'), self.bk_oe, groups=C)
        oo = F.conv2d(F.pad(x, [1, 2, 1, 2], mode='replicate'), self.bk_oo, groups=C)
        even_row = torch.stack([ee, eo], dim=-1).reshape(B, C, H, 2 * W)
        odd_row = torch.stack([oe, oo], dim=-1).reshape(B, C, H, 2 * W)
        return torch.stack([even_row, odd_row], dim=-2).reshape(B, C, 2 * H, 2 * W)

    def _upsample_step(self, source, guidance, target_h, target_w):
        small_guidance = F.adaptive_avg_pool2d(guidance, (target_h, target_w))
        return self.up(source, small_guidance)

    def forward(self, source, guidance):
        # 14->28->56->112->224
        s2 = self._upsample_step(source, guidance, 28, 28)
        s4 = self._upsample_step(s2, guidance, 56, 56)
        s8 = self._upsample_step(s4, guidance, 112, 112)
        s16 = self._upsample_step(s8, guidance, 224, 224)
        return self.fixup_proj(s16) * 0.1 + s16


# ============ Export Functions ============

def export_visual(clip_model):
    print("Exporting visual encoder...")
    visual_model = CLIPVisualEncoder(clip_model).npu().half().eval()
    dummy_image = torch.randn(1, 3, 224, 224, dtype=torch.float16, device='npu')
    output_path = os.path.join(OUTPUT_DIR, "clip_visual.onnx")

    with torch.no_grad():
        cls_token, tokens = visual_model(dummy_image)
        print(f"  cls_token: {cls_token.shape}, patch_tokens: {tokens.shape}")

        torch.onnx.export(
            visual_model,
            dummy_image,
            output_path,
            input_names=["image"],
            output_names=["cls_token", "patch_tokens"],
            opset_version=17,
            dynamic_axes=None,
        )
    print(f"  Saved: {output_path}")


def export_text(clip_model):
    print("Exporting text encoder...")
    text_model = CLIPTextEncoder(clip_model).npu().half().eval()
    dummy_text = tokenizer.tokenize(["a photo of a building"]).to(torch.int64).npu()
    output_path = os.path.join(OUTPUT_DIR, "clip_text.onnx")

    with torch.no_grad():
        out = text_model(dummy_text)
        print(f"  text_features: {out.shape}")

        torch.onnx.export(
            text_model,
            dummy_text,
            output_path,
            input_names=["text"],
            output_names=["text_features"],
            opset_version=17,
            dynamic_axes={"text": {0: "batch"}, "text_features": {0: "batch"}},
        )
    print(f"  Saved: {output_path}")


def export_upsampler_combined():
    """导出 JBU 为单个 ONNX (JBUUpsamplerWrapper: 4 级 upsample + fixup)。"""
    print("Exporting JBU upsampler (combined)...")
    feat_dim = 512
    upsampler = get_upsampler('jbu_one', feat_dim).npu().half().eval()
    ckpt_path = os.path.join(
        REPO_ROOT,
        'simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt')
    ckpt = torch.load(ckpt_path, map_location='npu', weights_only=False)['state_dict']
    weights_dict = {k[10:]: v for k, v in ckpt.items()}
    upsampler.load_state_dict(weights_dict, strict=True)

    wrapper = JBUUpsamplerWrapper(upsampler).npu().half().eval()

    dummy_source = torch.randn(1, 512, 14, 14, dtype=torch.float16, device='npu')
    dummy_guidance = torch.randn(1, 3, 224, 224, dtype=torch.float16, device='npu')
    output_path = os.path.join(OUTPUT_DIR, "jbu_upsampler.onnx")

    with torch.no_grad():
        out = wrapper(dummy_source, dummy_guidance)
        print(f"  input: source {dummy_source.shape}, guidance {dummy_guidance.shape}")
        print(f"  output: {out.shape}")

        torch.onnx.export(
            wrapper,
            (dummy_source, dummy_guidance),
            output_path,
            input_names=["source", "guidance"],
            output_names=["upsampled"],
            opset_version=17,
            dynamic_axes=None,
        )
    print(f"  Saved: {output_path}")


def export_upsampler_stages():
    """导出 JBU 为 4 个单级 OM + 1 个 fixup OM，避免单模型内存溢出。"""
    print("Exporting JBU upsampler (4 stages)...")
    feat_dim = 512
    upsampler = get_upsampler('jbu_one', feat_dim).npu().half().eval()
    ckpt_path = os.path.join(
        REPO_ROOT,
        'simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt')
    ckpt = torch.load(ckpt_path, map_location='npu', weights_only=False)['state_dict']
    weights_dict = {k[10:]: v for k, v in ckpt.items()}
    upsampler.load_state_dict(weights_dict, strict=True)

    up_module = upsampler.up.npu().half().eval()
    _inject_bicubic_buffers(up_module)

    stages = [(14, 28), (28, 56), (56, 112), (112, 224)]

    for idx, (src_hw, guid_hw) in enumerate(stages):
        dummy_src = torch.randn(1, 512, src_hw, src_hw, dtype=torch.float16, device='npu')
        dummy_guid = torch.randn(1, 3, guid_hw, guid_hw, dtype=torch.float16, device='npu')
        output_path = os.path.join(OUTPUT_DIR, f"jbu_stage{idx+1}.onnx")

        with torch.no_grad():
            out = up_module(dummy_src, dummy_guid)
            print(f"  stage{idx+1}: src[{src_hw}] + guid[{guid_hw}] -> {out.shape}")

            torch.onnx.export(
                up_module,
                (dummy_src, dummy_guid),
                output_path,
                input_names=["source", "guidance"],
                output_names=["upsampled"],
                opset_version=17,
                dynamic_axes=None,
            )
        print(f"  Saved: {output_path}")

    fixup = upsampler.fixup_proj.npu().half().eval()
    dummy_feat = torch.randn(1, 512, 224, 224, dtype=torch.float16, device='npu')
    output_path = os.path.join(OUTPUT_DIR, "jbu_fixup.onnx")
    with torch.no_grad():
        out = fixup(dummy_feat)
        print(f"  fixup: {dummy_feat.shape} -> {out.shape}")
        torch.onnx.export(
            fixup, dummy_feat, output_path,
            input_names=["features"], output_names=["projected"],
            opset_version=17, dynamic_axes=None,
        )
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    print("Loading CLIP ViT-B/16 (fp16, npu)...")
    clip_model = create_model('ViT-B/16', pretrained='openai', precision='fp16')
    clip_model.eval().npu()

    export_visual(clip_model)
    export_text(clip_model)
    export_upsampler_combined()
    export_upsampler_stages()
    print("\nDone! All ONNX files exported to models/onnx/")
