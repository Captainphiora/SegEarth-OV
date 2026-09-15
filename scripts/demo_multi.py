"""SegEarth-OV 统一多后端推理 demo。

用法:
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

    # PyTorch (默认)
    python scripts/demo_multi.py

    # ONNX (visual/text → onnxruntime, JBU → PyTorch)
    python scripts/demo_multi.py --backend onnx

    # ACL OM (纯 numpy, 无 PyTorch 依赖)
    python scripts/demo_multi.py --backend acl
"""

import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import argparse
import numpy as np
from PIL import Image


def _preprocess_numpy(img, size):
    """纯 numpy/PIL 预处理，等价于 torchvision ToTensor+Normalize+Resize。"""
    img_resized = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[np.newaxis, ...]


def _preprocess_torch(img, size):
    """torchvision 预处理（非 ACL 后端使用）。"""
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
        transforms.Resize((size, size)),
    ])(img).unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description="SegEarth-OV multi-backend demo")
    parser.add_argument("--backend", choices=["pytorch", "onnx", "acl", "acl_hybrid", "acl_stage"],
                        default="pytorch")
    parser.add_argument("--image", default="demo/oem_koeln_50.tif")
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--no-featup", action="store_true")
    parser.add_argument("--template", choices=["full", "sub"], default="full",
                        help="text prompt template: full (80) or sub (7)")
    parser.add_argument("--output", default=None,
                        help="output path (default: results/<backend>/seg_pred.png)")
    return parser.parse_args()


def main():
    args = parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils.session import Session, SessionConfig

    name_list = ['background', 'bareland,barren', 'grass', 'pavement', 'road',
                 'tree,forest', 'water,river', 'cropland', 'building,roof,house']

    cfg = SessionConfig(
        session_type=args.backend,
        feature_up=not args.no_featup,
        name_list=name_list,
        template=args.template,
    )

    print(f"Backend: {args.backend}")
    session = Session.from_config(cfg)

    img = Image.open(args.image).convert("RGB")

    if args.backend == "acl":
        img_tensor = _preprocess_numpy(img, args.size)
    else:
        img_tensor = _preprocess_torch(img, args.size)
        img_tensor = img_tensor.to("npu")

    print("Running inference...")
    import time as _time
    _t0 = _time.time()
    seg_pred = session.predict(img_tensor)
    _t1 = _time.time()
    print(f"Inference time: {_t1 - _t0:.3f}s (excluding model loading)")

    output = args.output or f"results/{args.backend}/seg_pred.png"
    os.makedirs(os.path.dirname(output), exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].axis("off")
    ax[1].imshow(seg_pred, cmap="viridis")
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    print(f"[OK] {output}")

    session.close()


if __name__ == "__main__":
    main()
