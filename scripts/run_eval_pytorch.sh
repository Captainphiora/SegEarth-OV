#!/bin/bash
# PyTorch eval on UDD5 — 一键执行
# 用法: bash scripts/run_eval_pytorch.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ========== 在这里修改配置 ==========
NPU_ID=3
TEMPLATE="full"
MAX_SAMPLES=5         # 0 = 全量
# ====================================

export NPU_ID

CONFIG="configs/cfg_udd5.py"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="work_logs/eval_pytorch_${TS}"
mkdir -p "$LOG_DIR"

# SHOW_DIR="$LOG_DIR/vis"
SHOW_DIR=""
LOG="$LOG_DIR/eval.log"

echo "PyTorch eval | NPU=$NPU_ID | template=$TEMPLATE | log_dir=$LOG_DIR"

python -u scripts/eval_acl.py \
    --config "$CONFIG" \
    --backend pytorch \
    --device-id 0 \
    --template "$TEMPLATE" \
    --max-samples "$MAX_SAMPLES" \
    --show-dir "$SHOW_DIR" \
    2>&1 | tee "$LOG"

echo ""
echo "日志目录: $LOG_DIR"
echo "  eval.log  — 完整输出"
echo "  vis/      — 可视化结果"
