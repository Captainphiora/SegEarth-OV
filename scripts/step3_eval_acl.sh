#!/bin/bash
# ============================================================
# Step 3: OM 模型 ACL 推理评估
# 使用编译好的 OM 模型在 NPU 上进行语义分割推理并计算精度
# 输入: models/om/*.om + UDD5 数据集
# 输出: mIoU/aAcc 精度指标 + 日志
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

NPU_ID="${NPU_ID:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-5}"
TEMPLATE="full"

export NPU_ID
source "$REPO_ROOT/scripts/env_npu.sh"

OM_DIR="$REPO_ROOT/models/om"
CONFIG="configs/cfg_udd5.py"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO_ROOT/logs/eval_acl_${TS}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/eval.log"

echo "========================================"
echo " Step 3: ACL Inference Evaluation"
echo " OM dir:      $OM_DIR"
echo " Config:      $CONFIG"
echo " NPU:         $NPU_ID"
echo " Max samples: $MAX_SAMPLES"
echo " Log dir:     $LOG_DIR"
echo "========================================"

for f in clip_visual.om clip_text.om jbu_upsampler.om; do
    if [ ! -f "$OM_DIR/$f" ]; then
        echo "ERROR: $OM_DIR/$f not found. Run step2 first."
        exit 1
    fi
done

python -u scripts/eval_acl.py \
    --config "$CONFIG" \
    --backend acl \
    --om-dir "$OM_DIR" \
    --device-id 0 \
    --template "$TEMPLATE" \
    --max-samples "$MAX_SAMPLES" \
    2>&1 | tee "$LOG"

echo ""
echo "日志: $LOG"
echo "========================================"
