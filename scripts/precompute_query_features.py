"""预计算 query features 并缓存到 .npy，避免推理时加载 text OM 占用显存。

用法:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    conda activate SegEarth

    # 全量 80 模板 (默认)
    python scripts/precompute_query_features.py

    # 精简 7 模板
    python scripts/precompute_query_features.py --template sub

    # 一次性生成两种缓存
    python scripts/precompute_query_features.py --template full
    python scripts/precompute_query_features.py --template sub
"""

import os, sys, argparse
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import hashlib, json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute query features for ACL inference")
    parser.add_argument("--template", choices=["full", "sub"], default="full",
                        help="full: 80 templates (higher accuracy), "
                             "sub: 7 templates (lower memory)")
    parser.add_argument("--om-dir", default="models/om")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing cache")
    return parser.parse_args()


def main():
    args = parse_args()

    name_list = ['background', 'bareland,barren', 'grass', 'pavement', 'road',
                 'tree,forest', 'water,river', 'cropland', 'building,roof,house']

    name_path = "./configs/my_name.txt"
    om_dir = args.om_dir

    os.makedirs(os.path.dirname(name_path) or ".", exist_ok=True)
    with open(name_path, "w") as f:
        f.write("\n".join(name_list))

    from segearth_segmentor import get_cls_idx
    query_words, query_idx = get_cls_idx(name_path)

    tpl_name = args.template
    cache_key = hashlib.md5(
        json.dumps(sorted(query_words)).encode()).hexdigest()[:12]
    cache_path = os.path.join(om_dir, f"query_features_{tpl_name}_{cache_key}.npy")

    if os.path.exists(cache_path) and not args.force:
        qf = np.load(cache_path)
        print(f"[OK] Cache already exists: {cache_path}")
        print(f"     shape={qf.shape}, dtype={qf.dtype}, template={tpl_name}")
        return

    from utils.session import _OmModel, _get_templates

    import acl
    acl.init()
    acl.rt.set_device(0)
    context, _ = acl.rt.create_context(0)

    text_om = os.path.join(om_dir, "clip_text.om")
    print(f"Loading text encoder: {text_om}")
    text_model = _OmModel(text_om, context)

    from open_clip import tokenizer
    templates = _get_templates(tpl_name)
    print(f"Using template={tpl_name} ({len(templates)} prompts)")

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
        print(f"  computed: {qw}")

    query_features = np.stack(features).astype(np.float16)
    np.save(cache_path, query_features)
    print(f"[OK] Saved {cache_path}  shape={query_features.shape}")

    text_model.close()
    acl.rt.destroy_context(context)
    acl.rt.reset_device(0)
    acl.finalize()


if __name__ == "__main__":
    main()
