#!/bin/bash
# scripts/run_profiling.sh — 项目级 Profiling 入口
#
# 用法:
#   bash scripts/run_profiling.sh "python -u scripts/eval_acl.py --config configs/cfg_udd5.py --backend acl --om-dir models/om --device-id 0 --template full --max-samples 1"
#   bash scripts/run_profiling.sh "python -u scripts/demo_multi.py --backend acl --size 448"
#
# 输出:
#   profiling_runs/<label>_<timestamp>/
#   ├── PROF_xxx/mindstudio_profiler_output/*.csv
#   ├── collect.log
#   ├── export.log
#   └── analysis.txt

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ========== 在这里修改配置 ==========
NPU_ID=0
LABEL="acl_eval"
# ====================================

NPU_PROFILE="/home/chenxinji/npu-tools/profiling/npu-profile.sh"

if [[ ! -f "$NPU_PROFILE" ]]; then
    echo "[ERROR] npu-profile.sh 未找到: $NPU_PROFILE"
    exit 1
fi

export NPU_ID
source "$SCRIPT_DIR/env_npu.sh"

APP_CMD=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --npu)   NPU_ID="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        --no-analysis) EXTRA_ARGS="$EXTRA_ARGS --no-analysis"; shift ;;
        --msprof-args) EXTRA_ARGS="$EXTRA_ARGS --msprof-args \"$2\""; shift 2 ;;
        --help|-h)
            echo "用法: bash $0 \"<command>\" [--npu 9] [--label name]"
            exit 0 ;;
        *)
            if [[ -z "$APP_CMD" ]]; then APP_CMD="$1"; else echo "未知参数: $1"; exit 1; fi
            shift ;;
    esac
done

if [[ -z "$APP_CMD" ]]; then
    echo "请提供要 profiling 的命令，例如:"
    echo "  bash $0 \"python -u scripts/eval_acl.py --config configs/cfg_udd5.py --backend acl --device-id 0 --max-samples 1\""
    exit 1
fi

bash "$NPU_PROFILE" "$APP_CMD" \
    --label "$LABEL" \
    --output-dir "profiling_runs/${LABEL}_$(date +%Y%m%d_%H%M%S)" \
    $EXTRA_ARGS
