"""SegEarth-OV 多后端推理会话管理器。

借鉴 qwen-ascend-llm 的 Session 模式:
  Session.from_config(cfg) → PyTorchSession / OnnxSession / AclSession

三个后端共享同一个 predict(img_tensor) 接口, 调用方无需关心底层实现。

用法:
    from utils.session import Session, SessionConfig

    cfg = SessionConfig(session_type="pytorch")   # or "onnx", "acl"
    session = Session.from_config(cfg)
    seg_pred = session.predict(img_tensor)         # [1, H, W] numpy
"""

import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class SessionConfig:
    session_type: str = "pytorch"

    clip_type: str = "CLIP"
    vit_type: str = "ViT-B/16"
    model_type: str = "SegEarth"
    ignore_residual: bool = True
    feature_up: bool = True
    jbu_model_name: str = "jbu_one"
    jbu_ckpt_path: str = "simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt"
    cls_token_lambda: float = -0.3
    prob_thd: float = 0.1
    logit_scale: float = 50.0
    slide_stride: int = 112
    slide_crop: int = 224
    bg_idx: int = 0

    name_path: str = "./configs/my_name.txt"
    name_list: Optional[List[str]] = None

    onnx_dir: str = "models/onnx"
    om_dir: str = "models/om"
    device_id: int = 0
    onnx_providers: Optional[List[str]] = None

    template: str = "full"
    debug_timing: bool = False


class Session:
    """推理会话基类。"""

    def __init__(self, config: SessionConfig):
        self.config = config

    def predict(self, img_tensor) -> np.ndarray:
        raise NotImplementedError

    def close(self):
        pass

    @staticmethod
    def from_config(config: SessionConfig) -> "Session":
        st = config.session_type.lower()
        if st == "pytorch":
            return PyTorchSession(config)
        elif st == "onnx":
            return OnnxSession(config)
        elif st == "acl":
            return AclSession(config)
        elif st == "acl_hybrid":
            return AclHybridSession(config)
        elif st == "acl_stage":
            return AclStageSession(config)
        else:
            raise ValueError(f"Unknown session_type: {config.session_type!r}, "
                             f"expected 'pytorch', 'onnx', 'acl', 'acl_hybrid', or 'acl_stage'")

    def __del__(self):
        self.close()


def _ensure_name_file(config: SessionConfig):
    """确保类名文件存在。"""
    if config.name_list is not None:
        os.makedirs(os.path.dirname(config.name_path) or ".", exist_ok=True)
        with open(config.name_path, "w") as f:
            f.write("\n".join(config.name_list))


TEMPLATE_REGISTRY = {
    "full": ("openai_imagenet_template", 80),
    "sub": ("sub_imagenet_template", 7),
}


def _get_templates(name: str):
    """按名称返回模板列表。"""
    if name not in TEMPLATE_REGISTRY:
        raise ValueError(
            f"Unknown template {name!r}, choose from {list(TEMPLATE_REGISTRY)}")
    attr_name, _ = TEMPLATE_REGISTRY[name]
    from prompts.imagenet_template import (
        openai_imagenet_template, sub_imagenet_template)
    return {"openai_imagenet_template": openai_imagenet_template,
            "sub_imagenet_template": sub_imagenet_template}[attr_name]


# ---------------------------------------------------------------------------
#  PyTorch 后端
# ---------------------------------------------------------------------------

class PyTorchSession(Session):

    def __init__(self, config: SessionConfig):
        super().__init__(config)
        _ensure_name_file(config)

        from segearth_segmentor import SegEarthSegmentation
        self.model = SegEarthSegmentation(
            clip_type=config.clip_type,
            vit_type=config.vit_type,
            model_type=config.model_type,
            ignore_residual=config.ignore_residual,
            feature_up=config.feature_up,
            feature_up_cfg=dict(
                model_name=config.jbu_model_name,
                model_path=config.jbu_ckpt_path),
            cls_token_lambda=config.cls_token_lambda,
            name_path=config.name_path,
            prob_thd=config.prob_thd,
            logit_scale=config.logit_scale,
            slide_stride=config.slide_stride,
            slide_crop=config.slide_crop,
            bg_idx=config.bg_idx,
        )
        print(f"[PyTorchSession] model loaded on NPU")

    def predict(self, img_tensor, ori_shape=None) -> np.ndarray:
        import torch
        from mmseg.structures import SegDataSample
        data_samples = None
        if ori_shape is not None:
            ds = SegDataSample()
            ds.set_metainfo(dict(
                ori_shape=ori_shape,
                img_shape=tuple(img_tensor.shape[2:]),
                pad_shape=tuple(img_tensor.shape[2:]),
                padding_size=[0, 0, 0, 0]))
            data_samples = [ds]
        with torch.no_grad():
            result = self.model.predict(img_tensor, data_samples=data_samples)
        if isinstance(result, list):
            return result[0].pred_sem_seg.data.cpu().numpy().squeeze(0)
        return result.data.cpu().numpy().squeeze(0)


# ---------------------------------------------------------------------------
#  ONNX 后端 (onnxruntime, 支持 CPU / CANNExecutionProvider)
# ---------------------------------------------------------------------------

class _OrtModel:
    """轻量 onnxruntime session 封装。"""

    def __init__(self, onnx_path, providers):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, opts, providers=providers)
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]
        actual = self.sess.get_providers()
        print(f"  [ORT] {os.path.basename(onnx_path)}: {actual[0]}")

    def __call__(self, *arrays):
        feed = {}
        for name, arr in zip(self.input_names, arrays):
            if hasattr(arr, 'cpu'):
                arr = arr.cpu().numpy()
            feed[name] = arr
        outs = self.sess.run(self.output_names, feed)
        return outs


class OnnxSession(Session):

    def __init__(self, config: SessionConfig):
        super().__init__(config)
        _ensure_name_file(config)

        providers = config.onnx_providers or ["CPUExecutionProvider"]
        onnx_dir = os.path.join(REPO_ROOT, config.onnx_dir)

        print(f"[OnnxSession] loading from {onnx_dir}")
        self.visual = _OrtModel(os.path.join(onnx_dir, "clip_visual.onnx"), providers)
        self.text = _OrtModel(os.path.join(onnx_dir, "clip_text.onnx"),
                              ["CPUExecutionProvider"])

        self._init_query_features(config)
        self._init_pytorch_pipeline(config)

    def _init_query_features(self, config):
        """用 ONNX text encoder 计算文本特征。"""
        import torch
        from open_clip import tokenizer
        from prompts.imagenet_template import openai_imagenet_template
        from segearth_segmentor import get_cls_idx

        query_words, self.query_idx = get_cls_idx(config.name_path)
        features = []
        for qw in query_words:
            tokens = tokenizer.tokenize(
                [t(qw) for t in openai_imagenet_template]).numpy().astype(np.int64)
            batch_feats = []
            for i in range(0, tokens.shape[0], 8):
                batch = tokens[i:i + 8]
                actual_n = batch.shape[0]
                if actual_n < 8:
                    batch = np.concatenate(
                        [batch, np.zeros((8 - actual_n, 77), dtype=np.int64)])
                out = self.text(batch)[0]
                batch_feats.append(out[:actual_n])
            feat = np.concatenate(batch_feats, axis=0).astype(np.float32).mean(axis=0)
            feat = feat / np.linalg.norm(feat)
            features.append(feat)
        self.query_features = torch.from_numpy(
            np.stack(features)).to(torch.float16).to("npu")
        print(f"  [ORT] text features: {self.query_features.shape}")

    def _init_pytorch_pipeline(self, config):
        """初始化 PyTorch pipeline, 但把 encode_image 替换为 ONNX。"""
        import torch
        import types
        from segearth_segmentor import SegEarthSegmentation

        self.model = SegEarthSegmentation(
            clip_type=config.clip_type,
            vit_type=config.vit_type,
            model_type=config.model_type,
            ignore_residual=config.ignore_residual,
            feature_up=config.feature_up,
            feature_up_cfg=dict(
                model_name=config.jbu_model_name,
                model_path=config.jbu_ckpt_path),
            cls_token_lambda=config.cls_token_lambda,
            name_path=config.name_path,
            prob_thd=config.prob_thd,
            logit_scale=config.logit_scale,
            slide_stride=config.slide_stride,
            slide_crop=config.slide_crop,
            bg_idx=config.bg_idx,
        )

        self.model.query_features = self.query_features

        visual_ort = self.visual

        def onnx_encode_image(self_net, img, model_type=None,
                              ignore_residual=False, output_cls_token=False, **kw):
            cls_token, patch_tokens = visual_ort(img)
            cls_token = torch.from_numpy(cls_token).to(img.device)
            patch_tokens = torch.from_numpy(patch_tokens).to(img.device)
            if output_cls_token:
                return cls_token, patch_tokens
            return patch_tokens

        self.model.net.encode_image = types.MethodType(onnx_encode_image, self.model.net)
        print(f"[OnnxSession] ready (visual encoder → ONNX, JBU → PyTorch)")

    def predict(self, img_tensor, ori_shape=None) -> np.ndarray:
        import torch
        from mmseg.structures import SegDataSample
        data_samples = None
        if ori_shape is not None:
            ds = SegDataSample()
            ds.set_metainfo(dict(
                ori_shape=ori_shape,
                img_shape=tuple(img_tensor.shape[2:]),
                pad_shape=tuple(img_tensor.shape[2:]),
                padding_size=[0, 0, 0, 0]))
            data_samples = [ds]
        with torch.no_grad():
            result = self.model.predict(img_tensor, data_samples=data_samples)
        if isinstance(result, list):
            return result[0].pred_sem_seg.data.cpu().numpy().squeeze(0)
        return result.data.cpu().numpy().squeeze(0)


# ---------------------------------------------------------------------------
#  ACL OM 后端 (纯 numpy pipeline, 无 PyTorch 依赖)
# ---------------------------------------------------------------------------

class _OmModel:
    """ACL OM 模型封装。

    每次 __call__ 前恢复 ACL context (防止 torch_npu 切换 context 导致 OM 执行异常)。
    """

    ACL_MEM_MALLOC_NORMAL_ONLY = 2
    ACL_MEMCPY_HOST_TO_DEVICE = 1
    ACL_MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self, om_path, acl_context=None):
        import acl
        self.acl = acl
        self._context = acl_context
        if acl_context is not None:
            acl.rt.set_context(acl_context)
        self.model_id, _ = acl.mdl.load_from_file(om_path)
        self.model_desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.model_desc, self.model_id)
        self.input_num = acl.mdl.get_num_inputs(self.model_desc)
        self.output_num = acl.mdl.get_num_outputs(self.model_desc)
        print(f"  [ACL] {os.path.basename(om_path)} loaded")

    def get_output_info(self, idx):
        size = self.acl.mdl.get_output_size_by_index(self.model_desc, idx)
        dims, _ = self.acl.mdl.get_output_dims(self.model_desc, idx)
        return size, dims["dims"]

    def __call__(self, *input_arrays):
        acl = self.acl
        if self._context is not None:
            acl.rt.set_context(self._context)
        dataset_in = acl.mdl.create_dataset()
        in_bufs = []
        for arr in input_arrays:
            arr = np.ascontiguousarray(arr)
            data = arr.tobytes()
            size = len(data)
            dev, _ = acl.rt.malloc(size, self.ACL_MEM_MALLOC_NORMAL_ONLY)
            acl.rt.memcpy(dev, size, acl.util.bytes_to_ptr(data), size,
                          self.ACL_MEMCPY_HOST_TO_DEVICE)
            buf = acl.create_data_buffer(dev, size)
            acl.mdl.add_dataset_buffer(dataset_in, buf)
            in_bufs.append((dev, buf))

        dataset_out = acl.mdl.create_dataset()
        out_bufs = []
        for i in range(self.output_num):
            size, _ = self.get_output_info(i)
            dev, _ = acl.rt.malloc(size, self.ACL_MEM_MALLOC_NORMAL_ONLY)
            buf = acl.create_data_buffer(dev, size)
            acl.mdl.add_dataset_buffer(dataset_out, buf)
            out_bufs.append((dev, size, buf))

        acl.mdl.execute(self.model_id, dataset_in, dataset_out)

        results = []
        for i, (dev, size, buf) in enumerate(out_bufs):
            host, _ = acl.rt.malloc_host(size)
            acl.rt.memcpy(host, size, dev, size, self.ACL_MEMCPY_DEVICE_TO_HOST)
            _, dims = self.get_output_info(i)
            data = acl.util.ptr_to_bytes(host, size)
            results.append(np.frombuffer(data, dtype=np.float16).reshape(dims).copy())
            acl.rt.free_host(host)
            acl.rt.free(dev)
            acl.destroy_data_buffer(buf)

        for dev, buf in in_bufs:
            acl.rt.free(dev)
            acl.destroy_data_buffer(buf)
        acl.mdl.destroy_dataset(dataset_in)
        acl.mdl.destroy_dataset(dataset_out)
        return results

    def close(self):
        self.acl.mdl.unload(self.model_id)
        self.acl.mdl.destroy_desc(self.model_desc)


class AclSession(Session):

    def __init__(self, config: SessionConfig):
        super().__init__(config)
        _ensure_name_file(config)
        import acl

        ret = acl.init()
        acl.rt.set_device(config.device_id)
        self._context, _ = acl.rt.create_context(config.device_id)
        self._acl = acl

        om_dir = os.path.join(REPO_ROOT, config.om_dir)

        print(f"[AclSession] loading from {om_dir}")
        ctx = self._context

        self._init_query_features_cached(config, om_dir, ctx)

        acl.rt.set_context(self._context)
        self.visual = _OmModel(os.path.join(om_dir, "clip_visual.om"), ctx)

        jbu_path = os.path.join(om_dir, "jbu_upsampler.om")
        self.upsampler = _OmModel(jbu_path, ctx) if os.path.exists(jbu_path) else None
        self.feature_up = config.feature_up and self.upsampler is not None

        self._cfg = config
        print(f"[AclSession] ready (featup={'ON' if self.feature_up else 'OFF'})")

    def _init_query_features_cached(self, config, om_dir, ctx):
        """加载或计算 query features，缓存到 .npy 避免重复加载 text OM。"""
        import hashlib, json
        from segearth_segmentor import get_cls_idx

        query_words, self.query_idx = get_cls_idx(config.name_path)

        tpl_name = config.template
        cache_key = hashlib.md5(
            json.dumps(sorted(query_words)).encode()).hexdigest()[:12]
        cache_path = os.path.join(
            om_dir, f"query_features_{tpl_name}_{cache_key}.npy")

        if os.path.exists(cache_path):
            print(f"[AclSession] loading cached query features: {cache_path}")
            self.query_features = np.load(cache_path)
        else:
            print(f"[AclSession] computing query features (template={tpl_name})...")
            text_model = _OmModel(os.path.join(om_dir, "clip_text.om"), ctx)
            self._compute_query_features(config, text_model, query_words)
            text_model.close()
            del text_model
            np.save(cache_path, self.query_features)
            print(f"[AclSession] query features cached to {cache_path}")

        self._qf = self.query_features.astype(np.float32)
        print(f"[AclSession] query features ready: {self.query_features.shape} "
              f"(template={tpl_name})")

    def _compute_query_features(self, config, text_model, query_words):
        from open_clip import tokenizer

        templates = _get_templates(config.template)
        features = []
        for qw in query_words:
            tokens = tokenizer.tokenize(
                [t(qw) for t in templates]).numpy().astype(np.int64)
            batch_feats = []
            for i in range(0, tokens.shape[0], 8):
                batch = tokens[i:i + 8]
                n = batch.shape[0]
                if n < 8:
                    batch = np.concatenate(
                        [batch, np.zeros((8 - n, 77), dtype=np.int64)])
                out = text_model(batch)[0]
                batch_feats.append(out[:n])
            feat = np.concatenate(batch_feats, 0).astype(np.float32).mean(0)
            feat /= np.linalg.norm(feat)
            features.append(feat)
        self.query_features = np.stack(features).astype(np.float16)

    def predict(self, img_tensor, ori_shape=None) -> np.ndarray:
        from PIL import Image

        if hasattr(img_tensor, "cpu"):
            img_np = img_tensor.cpu().numpy().astype(np.float16)
        else:
            img_np = np.asarray(img_tensor, dtype=np.float16)

        cfg = self._cfg
        logits = self._forward_slide(img_np, cfg.slide_stride, cfg.slide_crop)

        target_h, target_w = ori_shape if ori_shape else img_np.shape[2:]
        _, _, lh, lw = logits.shape
        if lh != target_h or lw != target_w:
            logits = self._bilinear_resize(logits, target_h, target_w)

        return self._postprocess(logits)

    def _forward_slide(self, img_np, stride, crop_size):
        _, _, h, w = img_np.shape
        nq = self.query_features.shape[0]
        h_grids = max(h - crop_size + stride - 1, 0) // stride + 1
        w_grids = max(w - crop_size + stride - 1, 0) // stride + 1
        preds = np.zeros((1, nq, h, w), dtype=np.float32)
        count = np.zeros((1, 1, h, w), dtype=np.float32)
        total_crops = h_grids * w_grids
        crop_idx = 0
        patch_size = 16
        _timing = self._cfg.debug_timing
        if _timing:
            import time as _time

        for hi in range(h_grids):
            for wi in range(w_grids):
                y1 = hi * stride; x1 = wi * stride
                y2 = min(y1 + crop_size, h); x2 = min(x1 + crop_size, w)
                y1 = max(y2 - crop_size, 0); x1 = max(x2 - crop_size, 0)
                crop = img_np[:, :, y1:y2, x1:x2].copy()

                cH, cW = crop.shape[2], crop.shape[3]
                l, r, t, b = 0, 0, 0, 0
                if cW % patch_size:
                    lr = patch_size - (cW % patch_size)
                    l = lr // 2; r = lr - l
                if cH % patch_size:
                    tb = patch_size - (cH % patch_size)
                    t = tb // 2; b = tb - t
                if l or r or t or b:
                    crop = np.pad(crop, ((0, 0), (0, 0), (t, b), (l, r)),
                                  mode='constant', constant_values=0)

                if _timing:
                    _ct0 = _time.time()
                logits = self._forward_feature(crop)
                if _timing:
                    _ct1 = _time.time()
                    print(f"  crop {crop_idx}/{total_crops}: {_ct1-_ct0:.3f}s")

                if l or r or t or b:
                    logits = logits[:, :, t:t + cH, l:l + cW]

                preds[:, :, y1:y2, x1:x2] += logits
                count[:, :, y1:y2, x1:x2] += 1
                crop_idx += 1

        return preds / count.clip(1)

    def _forward_feature(self, crop):
        _timing = self._cfg.debug_timing
        _, _, h, w = crop.shape
        self._acl.rt.set_context(self._context)

        if _timing:
            import time as _time
            _t0 = _time.time()
        cls_token, patch_tokens = self.visual(crop)
        if _timing:
            _t1 = _time.time()

        if self.feature_up:
            feat = patch_tokens.transpose(0, 2, 1).reshape(1, 512, h // 16, w // 16)
            up = self.upsampler(feat.astype(np.float16), crop.astype(np.float16))[0]
            image_features = up.reshape(1, 512, -1).transpose(0, 2, 1)
            sh, sw = h, w
        else:
            image_features = patch_tokens
            sh, sw = h // 16, w // 16
        if _timing:
            _t2 = _time.time()

        image_features = image_features.astype(np.float32)

        norm = np.linalg.norm(image_features, axis=-1, keepdims=True).clip(1e-12)
        imf = image_features / norm
        logits = np.einsum("bnd,cd->bnc", imf, self._qf)

        if self._cfg.cls_token_lambda != 0:
            ct = cls_token.astype(np.float32)
            ct_norm = np.linalg.norm(ct, axis=-1, keepdims=True).clip(1e-12)
            ct = ct / ct_norm
            ct_logits = (ct @ self._qf.T)[:, np.newaxis, :]
            logits = logits + ct_logits * self._cfg.cls_token_lambda
        if _timing:
            _t3 = _time.time()
            print(f"    visual={_t1-_t0:.3f}s  jbu={_t2-_t1:.3f}s  numpy={_t3-_t2:.3f}s")

        nq = logits.shape[-1]
        logits = logits.transpose(0, 2, 1).reshape(1, nq, sh, sw)

        if sh != h or sw != w:
            logits = self._bilinear_resize(logits, h, w)
        return logits

    @staticmethod
    def _bilinear_resize(logits, th, tw):
        from PIL import Image
        _, c, _, _ = logits.shape
        out = np.zeros((1, c, th, tw), dtype=np.float32)
        for i in range(c):
            pil = Image.fromarray(logits[0, i].astype(np.float32), mode="F")
            out[0, i] = np.array(pil.resize((tw, th), Image.BILINEAR))
        return out

    def _postprocess(self, logits):
        cfg = self._cfg
        seg = logits[0] * cfg.logit_scale
        e = np.exp(seg - seg.max(axis=0, keepdims=True))
        seg = e / e.sum(axis=0, keepdims=True)

        num_cls = max(self.query_idx) + 1
        num_queries = len(self.query_idx)
        if num_cls != num_queries:
            cls_logits = np.zeros((num_cls, *seg.shape[1:]), dtype=seg.dtype)
            for cls_id in range(num_cls):
                mask = [i for i, idx in enumerate(self.query_idx) if idx == cls_id]
                if mask:
                    cls_logits[cls_id] = seg[mask].max(axis=0)
            seg = cls_logits

        pred = seg.argmax(axis=0)
        pred[seg.max(axis=0) < cfg.prob_thd] = cfg.bg_idx
        return pred

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if hasattr(self, "visual"):
            try:
                self.visual.close()
            except Exception:
                pass
        if hasattr(self, "upsampler") and self.upsampler is not None:
            try:
                self.upsampler.close()
            except Exception:
                pass
        if hasattr(self, "_context"):
            try:
                self._acl.rt.destroy_context(self._context)
                if hasattr(self, "_cfg"):
                    self._acl.rt.reset_device(self._cfg.device_id)
                if "torch_npu" not in sys.modules:
                    self._acl.finalize()
            except (AttributeError, TypeError):
                pass
            del self._context
#  ACL per-stage 后端 (4 stage OM + fixup OM)
# ---------------------------------------------------------------------------

class AclStageSession(AclSession):
    """Per-stage JBU OM 推理：4 个独立 stage OM + 1 个 fixup OM。

    与 AclSession 的区别:
    - AclSession 用单个 jbu_upsampler.om 做完整 14→224 上采样
    - AclStageSession 用 4 个 jbu_stage{1-4}.om 逐级上采样 + jbu_fixup.om
    - 每级自行下采样 guidance (numpy bilinear)

    优势: 每级 OM 更小、显存峰值更低、可单级 profiling
    """

    _JBU_STAGES = [(14, 28), (28, 56), (56, 112), (112, 224)]

    def __init__(self, config: SessionConfig):
        Session.__init__(self, config)
        _ensure_name_file(config)
        import acl
        from PIL import Image as _PIL

        acl.init()
        acl.rt.set_device(config.device_id)
        self._context, _ = acl.rt.create_context(config.device_id)
        self._acl = acl

        om_dir = os.path.join(REPO_ROOT, config.om_dir)

        print(f"[AclStageSession] loading from {om_dir}")
        ctx = self._context

        print("[AclStageSession] loading text encoder for query feature computation...")
        text_model = _OmModel(os.path.join(om_dir, "clip_text.om"), ctx)
        self._init_query_features(config, text_model)
        text_model.close()
        del text_model
        print("[AclStageSession] text encoder unloaded")

        acl.rt.set_context(self._context)
        self.visual = _OmModel(os.path.join(om_dir, "clip_visual.om"), ctx)

        self.jbu_stages = []
        for i in range(1, 5):
            path = os.path.join(om_dir, f"jbu_stage{i}.om")
            self.jbu_stages.append(_OmModel(path, ctx))

        fixup_path = os.path.join(om_dir, "jbu_fixup.om")
        self.jbu_fixup = _OmModel(fixup_path, ctx)

        self.feature_up = config.feature_up
        self._cfg = config
        print(f"[AclStageSession] ready (featup={'ON' if self.feature_up else 'OFF'})")

    @staticmethod
    def _downsample_guidance(crop_fp16, target_h, target_w):
        """Numpy bilinear 下采样 guidance (与 F.adaptive_avg_pool2d 对齐)。"""
        from PIL import Image as _PIL
        _, c, _, _ = crop_fp16.shape
        out = np.zeros((1, c, target_h, target_w), dtype=np.float16)
        for ch in range(c):
            pil = _PIL.fromarray(crop_fp16[0, ch].astype(np.float32), mode="F")
            out[0, ch] = np.array(
                pil.resize((target_w, target_h), _PIL.BILINEAR),
                dtype=np.float16)
        return out

    def _forward_feature(self, crop):
        _, _, h, w = crop.shape
        cls_token, patch_tokens = self.visual(crop)

        if self.feature_up:
            source = patch_tokens.transpose(0, 2, 1).reshape(
                1, 512, h // 16, w // 16).astype(np.float16)
            crop_fp16 = crop.astype(np.float16)

            for idx, (src_hw, guid_hw) in enumerate(self._JBU_STAGES):
                guidance = self._downsample_guidance(
                    crop_fp16, guid_hw, guid_hw)
                source = self.jbu_stages[idx](source, guidance)[0]

            fixup_out = self.jbu_fixup(source)[0]
            up = fixup_out * np.float16(0.1) + source
            image_features = up.reshape(1, 512, -1).transpose(0, 2, 1)
            sh, sw = h, w
        else:
            image_features = patch_tokens
            sh, sw = h // 16, w // 16

        image_features = image_features.astype(np.float16)

        import torch
        import torch.nn.functional as F

        imf = torch.from_numpy(image_features).to("npu").half()
        imf = F.normalize(imf, dim=-1)
        logits = torch.einsum("bnd,cd->bnc", imf, self._qf_npu)

        if self._cfg.cls_token_lambda != 0:
            ct = torch.from_numpy(cls_token.astype(np.float16)).to("npu").half()
            ct = F.normalize(ct, dim=-1)
            logits = logits + (ct @ self._qf_npu.T).unsqueeze(1) * self._cfg.cls_token_lambda

        nq = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(1, nq, sh, sw)

        if sh != h or sw != w:
            logits = F.interpolate(logits, size=(h, w), mode="bilinear",
                                   align_corners=False)
        return logits.cpu().numpy()


# ---------------------------------------------------------------------------
#  ACL 混合后端 (Visual/Text OM + JBU PyTorch + Logits NPU)
# ---------------------------------------------------------------------------

class AclHybridSession(AclSession):
    """混合后端：Visual/Text 走 OM，JBU 留 PyTorch (原生 aclnnIm2col)。

    Visual OM: ~3.5ms/crop (ATC 图优化), JBU PyTorch: ~4ms/crop (原生 Im2col)
    """

    def __init__(self, config: SessionConfig):
        Session.__init__(self, config)
        _ensure_name_file(config)
        import acl
        import torch

        acl.init()
        acl.rt.set_device(config.device_id)
        self._context, _ = acl.rt.create_context(config.device_id)
        self._acl = acl

        om_dir = os.path.join(REPO_ROOT, config.om_dir)

        print(f"[AclHybridSession] loading OM from {om_dir}")
        ctx = self._context

        print("[AclHybridSession] loading text encoder for query feature computation...")
        text_model = _OmModel(os.path.join(om_dir, "clip_text.om"), ctx)
        self._init_query_features(config, text_model)
        text_model.close()
        del text_model
        print("[AclHybridSession] text encoder unloaded")

        acl.rt.set_context(self._context)
        self.visual = _OmModel(os.path.join(om_dir, "clip_visual.om"), ctx)
        self.upsampler = None
        self.feature_up = config.feature_up

        if config.feature_up:
            from simfeatup_dev.upsamplers import get_upsampler
            jbu = get_upsampler(config.jbu_model_name, 512).npu().half().eval()
            ckpt_path = os.path.join(REPO_ROOT, config.jbu_ckpt_path)
            ckpt = torch.load(ckpt_path, map_location='npu', weights_only=False)['state_dict']
            weights = {k[10:]: v for k, v in ckpt.items()}
            jbu.load_state_dict(weights, strict=True)
            self._jbu_pytorch = jbu
            print(f"  [PyTorch] JBU loaded on NPU")

        self._cfg = config
        print(f"[AclHybridSession] ready")

    def _forward_feature(self, crop):
        import torch
        import torch.nn.functional as F

        _, _, h, w = crop.shape
        cls_token, patch_tokens = self.visual(crop)

        if self.feature_up:
            feat_np = patch_tokens.transpose(0, 2, 1).reshape(1, 512, h // 16, w // 16)
            source = torch.from_numpy(feat_np).to("npu")
            guidance = torch.from_numpy(crop).to("npu")

            with torch.no_grad(), torch.npu.amp.autocast():
                up = self._jbu_pytorch(source, guidance)

            image_features = up.reshape(1, 512, -1).permute(0, 2, 1)
            sh, sw = h, w
        else:
            image_features = torch.from_numpy(patch_tokens).to("npu").float()
            sh, sw = h // 16, w // 16

        image_features = F.normalize(image_features.float(), dim=-1)
        logits = torch.einsum("bnd,cd->bnc", image_features, self._qf_npu)

        if self._cfg.cls_token_lambda != 0:
            ct = torch.from_numpy(cls_token.astype(np.float32)).to("npu")
            ct = F.normalize(ct, dim=-1)
            logits = logits + (ct @ self._qf_npu.T).unsqueeze(1) * self._cfg.cls_token_lambda

        nq = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(1, nq, sh, sw)

        if sh != h or sw != w:
            logits = F.interpolate(logits, size=(h, w), mode="bilinear",
                                   align_corners=False)
        return logits.cpu().numpy()
