#!/bin/bash
# ============================================================
# Step 1: 导出 ONNX 模型
# 使用 scripts/export_onnx.py (310B 优化版):
#   - ArgMax 用 ReduceMax+Equal+Where 替代 (310B1 无高优 ArgMaxV2)
#   - JBU bicubic 上采样用 depthwise conv2d 等价实现 (ATC 不支持 Resize half_pixel)
#   - Unfold 用 slice 累加替代 (避免 GatherV2 极慢)
# 产出在 models/onnx/ 下
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

ONNX_DIR="$REPO_ROOT/models/onnx"

echo "========================================"
echo " Step 1: PyTorch -> ONNX Export"
echo "========================================"

python -u scripts/export_onnx.py 2>&1 | tee "$REPO_ROOT/logs/step1_export_onnx.log"

echo ""
echo "ONNX files:"
ls -lh models/onnx/*.onnx 2>/dev/null || echo "ERROR: No ONNX files generated!"
