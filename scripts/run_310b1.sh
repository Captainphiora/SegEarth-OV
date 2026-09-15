#!/bin/bash
# ============================================================
# 310B1 OM 推理评估
# 用法: bash scripts/run_310b1.sh
#
# 使用前请修改下方 "用户配置" 区域的路径
# ============================================================
set -euo pipefail

# ==================== 用户配置 ====================
NPU_ID="${NPU_ID:-0}"                   # NPU 设备编号
CANN_HOME="${CANN_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"  # CANN 安装路径, 按实际修改
CONDA_ENV="${CONDA_ENV:-SegEarth}"       # Conda 环境名
OM_DIR="${OM_DIR:-models/om}"            # OM 文件目录 (含 clip_visual.om, clip_text.om, jbu_upsampler.om)
MAX_SAMPLES="${MAX_SAMPLES:-0}"          # 评估样本数, 0=全量
TEMPLATE="full"                          # 文本模板, full(80) 或 sub(7)
# ==================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

export NPU_ID CANN_HOME CONDA_ENV
export SKIP_NPU_CHECK=0
source "$REPO_ROOT/scripts/env_npu.sh"

CONFIG="configs/cfg_udd5.py"

echo "========================================"
echo " 310B1 ACL Inference"
echo " CANN:        $CANN_HOME"
echo " NPU:         $NPU_ID"
echo " OM dir:      $OM_DIR"
echo " Config:      $CONFIG"
echo " Max samples: $MAX_SAMPLES"
echo "========================================"

for f in clip_visual.om clip_text.om jbu_upsampler.om; do
    if [ ! -f "$OM_DIR/$f" ]; then
        echo "ERROR: $OM_DIR/$f not found."
        echo "请将 910 上编译好的 OM 文件放到 $OM_DIR/ 下"
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
    2>&1

echo "========================================"
echo " Done"
echo "========================================"
