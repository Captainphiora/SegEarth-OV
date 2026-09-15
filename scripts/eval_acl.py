"""SegEarth-OV 多后端评估脚本 (AclSession / PyTorchSession)。

支持 pytorch 和 acl 两种后端，统一走 Session 接口，计算 mIoU/aAcc/mAcc。

用法:
    source scripts/env_npu.sh

    # PyTorch eval
    python scripts/eval_acl.py --config configs/cfg_udd5.py --backend pytorch

    # ACL eval (指定 OM 目录)
    python scripts/eval_acl.py --config configs/cfg_udd5.py --backend acl \
        --om-dir models/om

    # 单张验证
    python scripts/eval_acl.py --config configs/cfg_udd5.py --backend acl \
        --om-dir models/om --max-samples 1
"""
import os, sys, time, argparse
import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import custom_visualizer  # noqa: F401 — 注册 TripletSegLocalVisualizer
import custom_datasets    # noqa: F401 — 注册数据集 (含 METAINFO)
from mmengine.config import Config
from mmseg.registry import DATASETS, VISUALIZERS


def load_dataset_paths(cfg):
    """从 mmseg config 中提取图片和标签路径列表。"""
    ds_cfg = cfg.test_dataloader.dataset
    data_root = ds_cfg.get('data_root', '')
    img_dir = os.path.join(data_root, ds_cfg.data_prefix.img_path)
    gt_dir = os.path.join(data_root, ds_cfg.data_prefix.seg_map_path)

    imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))])
    pairs = []
    for img_name in imgs:
        img_path = os.path.join(img_dir, img_name)
        gt_name = os.path.splitext(img_name)[0] + '.png'
        gt_path = os.path.join(gt_dir, gt_name)
        if not os.path.exists(gt_path):
            gt_name = img_name
            gt_path = os.path.join(gt_dir, gt_name)
        if os.path.exists(gt_path):
            pairs.append((img_path, gt_path))
    return pairs


def _keep_ratio_size(orig_w, orig_h, target_size):
    """计算 keep_ratio resize 后的尺寸 (与 MMSeg Resize keep_ratio=True 一致)。"""
    scale = min(target_size / orig_h, target_size / orig_w)
    new_w = int(orig_w * scale + 0.5)
    new_h = int(orig_h * scale + 0.5)
    return new_w, new_h


def preprocess_numpy(img, size):
    import cv2
    orig_w, orig_h = img.size
    new_w, new_h = _keep_ratio_size(orig_w, orig_h, size)
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_rgb = img_resized[:, :, ::-1].copy().astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr = (img_rgb - mean) / std
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...].astype(np.float16)


def preprocess_torch(img, size):
    import cv2, torch
    orig_w, orig_h = img.size
    new_w, new_h = _keep_ratio_size(orig_w, orig_h, size)
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_rgb = img_resized[:, :, ::-1].copy().astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    img_norm = (img_rgb - mean) / std
    return torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()


def compute_metrics(pred, gt, num_classes, ignore_index=255):
    """与 MMSeg IoUMetric 一致：返回逐类 intersection / union / gt_count 用于全局累积。"""
    valid = gt != ignore_index
    pred_v = pred[valid]
    gt_v = gt[valid]
    correct = (pred_v == gt_v).sum()
    total = valid.sum()
    intersections = np.zeros(num_classes, dtype=np.int64)
    unions = np.zeros(num_classes, dtype=np.int64)
    gt_counts = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        pred_c = pred_v == c
        gt_c = gt_v == c
        intersections[c] = (pred_c & gt_c).sum()
        unions[c] = (pred_c | gt_c).sum()
        gt_counts[c] = gt_c.sum()
    return {
        'intersections': intersections,
        'unions': unions,
        'gt_counts': gt_counts,
        'correct': int(correct),
        'total': int(total),
    }


def init_visualizer(cfg, show_dir):
    """初始化 MMSeg TripletSegLocalVisualizer。"""
    vis_cfg = cfg.get('visualizer', dict(type='TripletSegLocalVisualizer', name='visualizer'))
    vis_cfg['save_dir'] = show_dir
    visualizer = VISUALIZERS.build(vis_cfg)

    ds_type = cfg.test_dataloader.dataset.type
    ds_cls = DATASETS.module_dict.get(ds_type)
    if ds_cls and hasattr(ds_cls, 'METAINFO'):
        visualizer.dataset_meta = ds_cls.METAINFO
    return visualizer


def save_vis(visualizer, img_path, gt_seg, seg_pred, show_dir, step=0,
             max_side=720):
    """用 MMSeg visualizer 保存三栏可视化 (降分辨率, ~1.5s/张)。"""
    import cv2, torch
    import mmcv
    from mmseg.structures import SegDataSample
    from mmengine.structures import PixelData

    img = mmcv.imread(img_path, channel_order='rgb')
    h, w = img.shape[:2]

    if max_side > 0:
        sc = min(max_side / max(h, w), 1.0)
        if sc < 1.0:
            w, h = int(w * sc), int(h * sc)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    data_sample = SegDataSample()

    gt_resized = np.array(Image.fromarray(gt_seg.astype(np.uint8)).resize((w, h), Image.NEAREST))
    data_sample.gt_sem_seg = PixelData(data=torch.from_numpy(gt_resized[None].astype(np.int64)))

    pred_resized = np.array(Image.fromarray(seg_pred.astype(np.uint8)).resize((w, h), Image.NEAREST))
    data_sample.pred_sem_seg = PixelData(data=torch.from_numpy(pred_resized[None].astype(np.int64)))

    name = os.path.splitext(os.path.basename(img_path))[0]
    out_file = os.path.join(show_dir, f'{name}.png')
    visualizer.add_datasample(name, img, data_sample, out_file=out_file, step=step)


def parse_args():
    parser = argparse.ArgumentParser(description='SegEarth-OV multi-backend eval')
    parser.add_argument('--config', required=True)
    parser.add_argument('--backend', choices=['pytorch', 'acl', 'onnx'], default='acl')
    parser.add_argument('--om-dir', default='models/om')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--size', type=int, default=448)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--show-dir', default='',
                        help='保存可视化结果的目录 (空=不保存)')
    parser.add_argument('--template', choices=['full', 'sub'], default='full',
                        help='text prompt template: full (80) or sub (7)')
    parser.add_argument('--debug-timing', action='store_true',
                        help='print per-crop timing breakdown')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    name_path = cfg.model.get('name_path', '')

    from segearth_segmentor import get_cls_idx
    query_words, query_idx = get_cls_idx(name_path)
    num_classes = max(query_idx) + 1

    name_list = []
    with open(name_path) as f:
        for line in f:
            line = line.strip()
            if line:
                name_list.append(line)

    print("=" * 60)
    print(f"  SegEarth-OV Eval")
    print(f"  Backend:  {args.backend}")
    print(f"  Config:   {args.config}")
    print(f"  OM dir:   {args.om_dir}" if args.backend == 'acl' else "")
    print(f"  Device:   NPU {args.device_id}")
    print(f"  Size:     {args.size}")
    print(f"  Classes:  {num_classes} ({name_list})")
    print("=" * 60)

    from utils.session import Session, SessionConfig
    session_cfg = SessionConfig(
        session_type=args.backend,
        om_dir=args.om_dir,
        device_id=args.device_id,
        name_list=name_list,
        prob_thd=cfg.model.get('prob_thd', 0.1),
        logit_scale=cfg.model.get('logit_scale', 50.0),
        bg_idx=cfg.model.get('bg_idx', 0),
        cls_token_lambda=cfg.model.get('cls_token_lambda', -0.3),
        template=args.template,
        debug_timing=args.debug_timing,
    )

    print("\nLoading model...")
    t_load = time.time()
    session = Session.from_config(session_cfg)
    t_load = time.time() - t_load
    print(f"Model loaded in {t_load:.2f}s")

    print("\nLoading dataset...")
    pairs = load_dataset_paths(cfg)
    total = len(pairs)
    if args.max_samples > 0:
        total = min(total, args.max_samples)
    print(f"Dataset: {cfg.test_dataloader.dataset.type}, samples: {total}")

    visualizer = None
    if args.show_dir:
        visualizer = init_visualizer(cfg, args.show_dir)
        print(f"Visualization: {args.show_dir}")

    all_metrics = []
    infer_times = []

    print("\nRunning evaluation...")
    for idx in range(total):
        img_path, gt_path = pairs[idx]
        gt_seg = np.array(Image.open(gt_path), dtype=np.int32)
        gt_h, gt_w = gt_seg.shape

        img = Image.open(img_path).convert('RGB')
        ori_shape = (gt_h, gt_w)

        if args.backend == 'acl':
            img_input = preprocess_numpy(img, args.size)
        else:
            import torch
            img_input = preprocess_torch(img, args.size).to(f"npu:{args.device_id}")

        t0 = time.time()
        seg_pred = session.predict(img_input, ori_shape=ori_shape)
        t_infer = time.time() - t0
        infer_times.append(t_infer)

        if seg_pred.shape != gt_seg.shape:
            seg_pred_pil = Image.fromarray(seg_pred.astype(np.uint8))
            seg_pred = np.array(seg_pred_pil.resize((gt_w, gt_h), Image.NEAREST))

        metrics = compute_metrics(seg_pred, gt_seg, num_classes)
        all_metrics.append(metrics)

        if visualizer is not None:
            save_vis(visualizer, img_path, gt_seg, seg_pred, args.show_dir, step=idx)

        if (idx + 1) % 10 == 0 or idx == total - 1 or total <= 5:
            tot_inter = sum(m['intersections'] for m in all_metrics)
            tot_union = sum(m['unions'] for m in all_metrics)
            ious = tot_inter / np.maximum(tot_union, 1)
            miou = np.nanmean(ious) * 100
            corr = sum(m['correct'] for m in all_metrics)
            tot = sum(m['total'] for m in all_metrics)
            aacc = corr / tot * 100 if tot > 0 else 0
            avg_time = np.mean(infer_times)
            print(f"  [{idx+1:3d}/{total}] mIoU={miou:5.2f}%  aAcc={aacc:5.2f}%  "
                  f"infer={t_infer:.3f}s  avg={avg_time:.3f}s")

    session.close()

    tot_inter = sum(m['intersections'] for m in all_metrics)
    tot_union = sum(m['unions'] for m in all_metrics)
    tot_gt = sum(m['gt_counts'] for m in all_metrics)
    ious = tot_inter / np.maximum(tot_union, 1).astype(np.float64)
    accs = tot_inter / np.maximum(tot_gt, 1).astype(np.float64)
    miou = np.nanmean(ious) * 100
    macc = np.nanmean(accs) * 100
    total_correct = sum(m['correct'] for m in all_metrics)
    total_pixels = sum(m['total'] for m in all_metrics)
    aacc = total_correct / total_pixels * 100

    total_infer = sum(infer_times)
    avg_infer = np.mean(infer_times)
    throughput = total / total_infer if total_infer > 0 else 0

    print("\n" + "=" * 60)
    print(f"  Results: {args.config} [{args.backend}]")
    print("=" * 60)
    print(f"  aAcc:       {aacc:.2f}%")
    print(f"  mIoU:       {miou:.2f}%")
    print(f"  mAcc:       {macc:.2f}%")
    print(f"  Per-class IoU:")
    for c in range(num_classes):
        print(f"    {name_list[c]:20s}  IoU={ious[c]*100:5.2f}%  Acc={accs[c]*100:5.2f}%")
    print(f"\n  Timing:")
    print(f"    Model load:    {t_load:.2f}s")
    print(f"    Total infer:   {total_infer:.2f}s ({total} images)")
    print(f"    Avg per image: {avg_infer:.3f}s")
    print(f"    Throughput:    {throughput:.2f} img/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
