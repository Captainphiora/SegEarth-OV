#!/bin/bash
# ============================================================
# Step 0: PyTorch 基线评估 (原始 eval.py, MMEngine Runner)
# 用于与 ACL 后端对比精度
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

export NPU_ID="${NPU_ID:-4}"
export SKIP_NPU_CHECK=1
source "$REPO_ROOT/scripts/env_npu.sh"

CONFIG="configs/cfg_udd5.py"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO_ROOT/logs/eval_pytorch_${TS}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/eval.log"

echo "========================================"
echo " Step 0: PyTorch Baseline (eval.py)"
echo " Config: $CONFIG"
echo " NPU:    $NPU_ID"
echo " Log:    $LOG_DIR"
echo "========================================"

python -u eval.py \
    --config "$CONFIG" \
    --work-dir "$LOG_DIR" \
    2>&1 | tee "$LOG"

echo ""
echo "日志: $LOG"
echo "========================================"
