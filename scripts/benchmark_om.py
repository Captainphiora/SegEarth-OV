"""OM 推理综合 Benchmark: PyTorch baseline vs ACL combined vs ACL per-stage。

输出:
  - 终端汇总表
  - work_logs/benchmark_YYYYMMDD_HHMMSS.log 完整日志
  - results/<backend>/seg_pred.png 分割结果 (用于精度对比)

用法:
    source scripts/env_npu.sh
    python scripts/benchmark_om.py                  # 全量 (pytorch + acl + acl_stage)
    python scripts/benchmark_om.py --backends acl acl_stage  # 只跑指定后端
"""
import os, sys, time, json, argparse, subprocess, re, datetime
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
from PIL import Image
from torchvision import transforms

WARMUP = 1
REPEAT = 3
IMG_PATH = "demo/oem_koeln_50.tif"
IMG_SIZE = 448

LOG_DIR = os.path.join(REPO_ROOT, "work_logs")
os.makedirs(LOG_DIR, exist_ok=True)

NAME_LIST = ['background', 'bareland,barren', 'grass', 'pavement', 'road',
             'tree,forest', 'water,river', 'cropland', 'building,roof,house']


class Logger:
    def __init__(self, path):
        self.path = path
        self.lines = []
        self._fh = open(path, "w")

    def log(self, msg=""):
        print(msg)
        self._fh.write(msg + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


def get_npu_memory(device_id=0):
    try:
        out = subprocess.check_output(["npu-smi", "info"], text=True, timeout=5)
        matches = re.findall(r'(\d+)\s*/\s*(\d+)\s*\|?\s*$', out, re.MULTILINE)
        chip = 0
        for usage_s, cap_s in matches:
            cap = int(cap_s)
            if cap > 1000:
                if chip == device_id:
                    return int(usage_s), cap
                chip += 1
    except Exception:
        pass
    return 0, 0


def prepare_image():
    img = Image.open(IMG_PATH).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ])
    return transform(img).unsqueeze(0)


def save_seg_pred(pred, backend):
    out_dir = os.path.join("results", backend)
    os.makedirs(out_dir, exist_ok=True)
    from matplotlib import colormaps
    cmap = colormaps.get_cmap("viridis")
    num_cls = int(pred.max()) + 1
    colors = (cmap(np.linspace(0, 1, max(num_cls, 2)))[:, :3] * 255).astype(np.uint8)
    rgb = colors[pred]
    Image.fromarray(rgb).save(os.path.join(out_dir, "seg_pred.png"))


def compute_accuracy(pred, ref_pred):
    if pred.shape != ref_pred.shape:
        return {"pixel_match": 0.0, "note": f"shape mismatch {pred.shape} vs {ref_pred.shape}"}
    total = pred.size
    match = (pred == ref_pred).sum()
    return {
        "pixel_match_pct": round(match / total * 100, 2),
        "diff_pixels": int(total - match),
        "total_pixels": int(total),
    }


def benchmark_one(backend, img_tensor, log: Logger, warmup=WARMUP, repeat=REPEAT):
    from utils.session import Session, SessionConfig

    log.log(f"\n{'='*60}")
    log.log(f"  Backend: {backend}")
    log.log(f"{'='*60}")

    hbm_before, hbm_cap = get_npu_memory()
    log.log(f"  HBM before:       {hbm_before} / {hbm_cap} MB")

    cfg = SessionConfig(session_type=backend, name_list=NAME_LIST)
    t0 = time.time()
    try:
        session = Session.from_config(cfg)
    except Exception as exc:
        log.log(f"  [ERROR] Failed to init: {exc}")
        return None
    load_time = time.time() - t0

    hbm_after_load, _ = get_npu_memory()
    log.log(f"  Model load:       {load_time:.2f}s")
    log.log(f"  HBM after load:   {hbm_after_load} MB (+{hbm_after_load - hbm_before} MB)")

    inp = img_tensor
    if backend not in ("acl", "acl_stage"):
        inp = img_tensor.to("npu")

    log.log(f"  Warmup ({warmup} runs)...")
    for _ in range(warmup):
        try:
            pred = session.predict(inp)
        except Exception as exc:
            log.log(f"  [ERROR] Warmup failed: {exc}")
            session.close()
            return None

    hbm_after_infer, _ = get_npu_memory()
    log.log(f"  HBM after infer:  {hbm_after_infer} MB (+{hbm_after_infer - hbm_before} MB)")

    log.log(f"  Timed runs ({repeat})...")
    times = []
    for i in range(repeat):
        t0 = time.time()
        pred = session.predict(inp)
        elapsed = time.time() - t0
        times.append(elapsed)
        log.log(f"    run {i+1}: {elapsed:.3f}s")

    save_seg_pred(pred, backend)

    result = {
        "backend": backend,
        "load_time_s": round(load_time, 3),
        "infer_mean_s": round(np.mean(times), 3),
        "infer_std_s": round(np.std(times), 3),
        "infer_times_s": [round(t, 3) for t in times],
        "hbm_before_mb": hbm_before,
        "hbm_load_mb": hbm_after_load,
        "hbm_infer_mb": hbm_after_infer,
        "hbm_delta_load_mb": hbm_after_load - hbm_before,
        "hbm_delta_infer_mb": hbm_after_infer - hbm_before,
    }

    log.log(f"  Inference:        {result['infer_mean_s']:.3f}s ± {result['infer_std_s']:.3f}s")
    log.log(f"  HBM Δ(load):      +{result['hbm_delta_load_mb']} MB")
    log.log(f"  HBM Δ(infer):     +{result['hbm_delta_infer_mb']} MB")

    return result, pred, session


def print_summary(results, log: Logger):
    log.log(f"\n{'='*72}")
    log.log("  综合对比")
    log.log(f"{'='*72}")

    headers = [""] + [r["backend"] for r in results]
    rows = [
        ["Load time"] + [f"{r['load_time_s']:.2f}s" for r in results],
        ["Inference"] + [f"{r['infer_mean_s']:.3f}s ± {r['infer_std_s']:.3f}s" for r in results],
        ["HBM Δ(load)"] + [f"+{r['hbm_delta_load_mb']}MB" for r in results],
        ["HBM Δ(infer)"] + [f"+{r['hbm_delta_infer_mb']}MB" for r in results],
    ]

    if len(results) > 1 and "accuracy" in results[1]:
        for r in results[1:]:
            acc = r.get("accuracy", {})
            rows.append(
                ["Pixel match"] +
                ["baseline"] +
                [f"{acc.get('pixel_match_pct', '?')}%" for r2 in results[1:]
                 if r2.get("accuracy")]
            )
            break

    col_widths = [max(len(str(row[i])) for row in [headers] + rows)
                  for i in range(len(headers))]
    fmt = "  " + "  ".join(f"{{:>{w}}}" for w in col_widths)
    log.log(fmt.format(*headers))
    log.log("  " + "-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows:
        log.log(fmt.format(*row))
    log.log(f"{'='*72}")


def run_single_backend(backend, warmup, repeat, json_out_path):
    """单后端入口 — 供子进程调用。"""
    log = Logger("/dev/null")
    log.log = print

    img_tensor = prepare_image()
    out = benchmark_one(backend, img_tensor, log, warmup, repeat)
    if out is None:
        sys.exit(1)
    result, pred, session = out

    save_seg_pred(pred, backend)

    pred_path = json_out_path.replace(".json", f"_pred_{backend}.npy")
    np.save(pred_path, pred)
    result["pred_path"] = pred_path

    with open(json_out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+",
                        default=["pytorch", "acl", "acl_stage"],
                        help="Backends to benchmark")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--repeat", type=int, default=REPEAT)
    parser.add_argument("--_single", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_json_out", type=str, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._single:
        run_single_backend(args._single, args.warmup, args.repeat, args._json_out)
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"benchmark_{ts}.log")
    log = Logger(log_path)

    log.log(f"SegEarth-OV OM Benchmark — {ts}")
    log.log(f"Image: {IMG_PATH} ({IMG_SIZE}×{IMG_SIZE})")
    log.log(f"Backends: {args.backends}")
    log.log(f"Warmup: {args.warmup}, Repeat: {args.repeat}")

    tmp_dir = os.path.join(LOG_DIR, f"_bench_{ts}")
    os.makedirs(tmp_dir, exist_ok=True)

    all_results = []
    preds = {}

    for backend in args.backends:
        log.log(f"\n>>> Launching {backend} in subprocess...")
        json_tmp = os.path.join(tmp_dir, f"{backend}.json")

        cmd = [sys.executable, __file__,
               "--_single", backend,
               "--_json_out", json_tmp,
               "--warmup", str(args.warmup),
               "--repeat", str(args.repeat)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        log.log(proc.stdout.rstrip() if proc.stdout else "")
        if proc.stderr:
            for line in proc.stderr.strip().split("\n"):
                if "Warning" not in line and "FutureWarning" not in line:
                    log.log(f"  [stderr] {line}")

        if proc.returncode != 0 and not os.path.exists(json_tmp):
            log.log(f"  [SKIP] {backend} failed (rc={proc.returncode})")
            continue
        if not os.path.exists(json_tmp):
            log.log(f"  [SKIP] {backend} produced no result JSON")
            continue
        if proc.returncode != 0:
            log.log(f"  [WARN] {backend} exited with rc={proc.returncode} (result saved, likely cleanup issue)")

        with open(json_tmp) as f:
            result = json.load(f)
        all_results.append(result)

        pred_path = result.get("pred_path")
        if pred_path and os.path.exists(pred_path):
            preds[backend] = np.load(pred_path)

    ref_backend = args.backends[0] if args.backends[0] in preds else None
    if ref_backend:
        ref_pred = preds[ref_backend]
        for r in all_results:
            b = r["backend"]
            if b == ref_backend:
                r["accuracy"] = {"note": "baseline"}
            elif b in preds:
                r["accuracy"] = compute_accuracy(preds[b], ref_pred)
                log.log(f"\n  Accuracy ({b} vs {ref_backend}): {r['accuracy']}")

    if all_results:
        print_summary(all_results, log)

    json_path = log_path.replace(".log", ".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.log(f"\nJSON: {json_path}")
    log.log(f"Log:  {log_path}")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    log.close()


if __name__ == "__main__":
    main()
