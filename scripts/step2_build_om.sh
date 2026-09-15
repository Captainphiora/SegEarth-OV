#!/bin/bash
# ============================================================
# Step 2: ONNX -> OM 编译
# 使用 ATC 将三个 ONNX 模型编译为昇腾 OM 离线模型
# 输入: models/onnx/*.onnx (step1 已生成)
# 输出: models/om/*.om
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

ONNX_DIR="$REPO_ROOT/models/onnx"
OM_DIR="$REPO_ROOT/models/om"

SOC_VERSION="${SOC_VERSION:-Ascend910_9382}"

export NPU_ID="${NPU_ID:-4}"
export SKIP_NPU_CHECK=1
source "$REPO_ROOT/scripts/env_npu.sh"

mkdir -p "$OM_DIR"

echo "========================================"
echo " Step 2: ATC ONNX -> OM Compilation"
echo " SOC:       $SOC_VERSION"
echo " ONNX dir:  $ONNX_DIR"
echo " OM dir:    $OM_DIR"
echo "========================================"

for f in clip_visual.onnx clip_text.onnx jbu_upsampler.onnx; do
    if [ ! -f "$ONNX_DIR/$f" ]; then
        echo "ERROR: $ONNX_DIR/$f not found. Run step1 first."
        exit 1
    fi
done

echo ""
echo "[1/3] clip_visual.onnx -> clip_visual.om"
echo "      input: image [1, 3, 224, 224]"
atc --model="$ONNX_DIR/clip_visual.onnx" \
    --framework=5 \
    --output="$OM_DIR/clip_visual" \
    --soc_version=${SOC_VERSION} \
    --input_shape="image:1,3,224,224" \
    --input_format=NCHW \
    --log=error \
    2>&1 | tee "$REPO_ROOT/logs/step2_atc_visual.log"
echo "  -> clip_visual.om OK"

echo ""
echo "[2/3] clip_text.onnx -> clip_text.om"
echo "      input: text [8, 77] int64"
atc --model="$ONNX_DIR/clip_text.onnx" \
    --framework=5 \
    --output="$OM_DIR/clip_text" \
    --soc_version=${SOC_VERSION} \
    --input_shape="text:8,77" \
    --log=error \
    2>&1 | tee "$REPO_ROOT/logs/step2_atc_text.log"
echo "  -> clip_text.om OK"

echo ""
echo "[3/3] jbu_upsampler.onnx -> jbu_upsampler.om"
echo "      input: source [1, 512, 14, 14], guidance [1, 3, 224, 224]"
atc --model="$ONNX_DIR/jbu_upsampler.onnx" \
    --framework=5 \
    --output="$OM_DIR/jbu_upsampler" \
    --soc_version=${SOC_VERSION} \
    --input_shape="source:1,512,14,14;guidance:1,3,224,224" \
    --input_format=NCHW \
    --log=error \
    2>&1 | tee "$REPO_ROOT/logs/step2_atc_upsampler.log"
echo "  -> jbu_upsampler.om OK"

echo ""
echo "========================================"
echo " All OM models compiled successfully!"
echo "========================================"
ls -lh "$OM_DIR"/*.om
