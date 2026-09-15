#!/bin/bash
# ONNX 导出 + ATC 编译 (310B1, 自定义 ArgMax 替代)
# 用法: bash scripts/build_om_310b.sh
# 注意: 可在 910 服务器上交叉编译, ATC 只需 --soc_version=Ascend310B1
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.0.0/opp

# SOC="Ascend310B1"
SOC="Ascend910_9382"
ONNX_DIR="$REPO_ROOT/models/onnx"
OM_DIR="$REPO_ROOT/models/om"
LOG_DIR="$OM_DIR/logs"

mkdir -p "$ONNX_DIR" "$OM_DIR" "$LOG_DIR"

echo "============================================"
echo "  Build OM for $SOC (custom ArgMax)"
echo "  ONNX -> $ONNX_DIR"
echo "  OM   -> $OM_DIR"
echo "============================================"

# Step 1: ONNX 导出 (--no-native-argmax 使用自定义 ArgMax 替代)
echo ""
echo "[Step 1] Exporting ONNX (--no-native-argmax)..."
rm -f "$ONNX_DIR"/*.onnx
python "$SCRIPT_DIR/export_onnx.py" --no-native-argmax 2>&1 | tee "$LOG_DIR/onnx_export.log"

# Step 2: ATC 编译
echo ""
echo "[Step 2] ATC compile..."
for model in clip_visual jbu_upsampler; do
    echo "--- $model ---"
    atc --model="$ONNX_DIR/${model}.onnx" \
        --framework=5 \
        --output="$OM_DIR/${model}" \
        --soc_version="$SOC" \
        --log=info 2>&1 | tee "$LOG_DIR/${model}_atc.log"
    echo ""
done

echo "--- clip_text ---"
atc --model="$ONNX_DIR/clip_text.onnx" \
    --framework=5 \
    --output="$OM_DIR/clip_text" \
    --soc_version="$SOC" \
    --input_shape="text:8,77" \
    --log=info 2>&1 | tee "$LOG_DIR/clip_text_atc.log"

# Step 3: 验证产物
echo ""
echo "============================================"
echo "  Build complete"
echo "============================================"
ls -lh "$OM_DIR"/*.om
echo ""
echo "Logs saved to $LOG_DIR/"
