#!/bin/bash
# ACL eval on UDD5 — 一键执行
# 用法: bash scripts/run_eval_acl.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ========== 在这里修改配置 ==========
NPU_ID=0
TEMPLATE="full"
# MAX_SAMPLES=1          # 0 = 全量
# MAX_SAMPLES=0          # 0 = 全量
MAX_SAMPLES=5          # 0 = 全量
# ====================================

export NPU_ID
source "$SCRIPT_DIR/env_npu.sh"

CONFIG="configs/cfg_udd5.py"
# OM_DIR="models/om"
OM_DIR="models/om"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="work_logs/eval_acl_${TS}"
mkdir -p "$LOG_DIR"

# SHOW_DIR="$LOG_DIR/vis"
SHOW_DIR=""
LOG="$LOG_DIR/eval.log"

echo "ACL eval | NPU=$NPU_ID | template=$TEMPLATE | log_dir=$LOG_DIR"

python -u scripts/eval_acl.py \
    --config "$CONFIG" \
    --backend acl \
    --om-dir "$OM_DIR" \
    --device-id 0 \
    --template "$TEMPLATE" \
    --max-samples "$MAX_SAMPLES" \
    --show-dir "$SHOW_DIR" \
    2>&1 | tee "$LOG"

echo ""
echo "日志目录: $LOG_DIR"
echo "  eval.log  — 完整输出"
echo "  vis/      — 可视化结果"
