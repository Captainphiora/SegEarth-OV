#!/usr/bin/env bash
# Common Ascend/CANN + conda environment for SegEarth-OV (910 branch).
# Source this file from the repository root or from a wrapper script:
#   source ./env_npu.sh
# Optional overrides: CONDA_ENV, CANN_HOME, NPU_ID, HF_MIRROR.

# Do not use CANN's set_env.sh here: on this host it points to the obsolete
# ${CANN_HOME}/cann-9.0.0 layout.  The installed toolkit uses ${CANN_HOME}
# directly, so all paths are made explicit below.
if [[ -z "${SEG_EARTH_ROOT:-}" ]]; then
    SEG_EARTH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export SEG_EARTH_ROOT

CONDA_ENV="${CONDA_ENV:-SegEarth}"
CANN_HOME="${CANN_HOME:-/usr/local/Ascend/cann-9.0.0}"
NPU_ID="${NPU_ID:-0}"
HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"
export CONDA_ENV CANN_HOME NPU_ID HF_MIRROR

# Activate conda when this file is sourced from a regular shell.  A caller may
# set SKIP_CONDA_ACTIVATE=1 when it already activated the desired environment.
if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
    for conda_root in "${CONDA_ROOT:-}" /root/miniconda3 "${HOME:-}/miniconda3" /opt/conda; do
        if [[ -n "$conda_root" && -f "$conda_root/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1090
            source "$conda_root/etc/profile.d/conda.sh"
            conda activate "$CONDA_ENV"
            break
        fi
    done
fi

if [[ ! -d "$CANN_HOME" ]]; then
    echo "[env_npu] CANN_HOME does not exist: $CANN_HOME" >&2
    return 1 2>/dev/null || exit 1
fi

export ASCEND_HOME_PATH="$CANN_HOME"
export ASCEND_TOOLKIT_HOME="$CANN_HOME"
export ASCEND_OPP_PATH="$CANN_HOME/opp"
export ASCEND_AICPU_PATH="$CANN_HOME"
export ASCEND_RT_VISIBLE_DEVICES="$NPU_ID"
export HF_ENDPOINT="$HF_MIRROR"

export PATH="$CANN_HOME/compiler/bin:$CANN_HOME/aarch64-linux/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CANN_HOME/aarch64-linux/lib64:$CANN_HOME/runtime/lib64:$CANN_HOME/compiler/lib64:$CANN_HOME/opp/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CANN_HOME/python/site-packages:$CANN_HOME/opp/built-in/op_impl/ai_core/tbe:$SEG_EARTH_ROOT:${PYTHONPATH:-}"

if [[ "${SKIP_NPU_CHECK:-0}" != "1" ]]; then
    if ! command -v python >/dev/null 2>&1; then
        echo "[env_npu] python is not available after activating '$CONDA_ENV'" >&2
        return 1 2>/dev/null || exit 1
    fi
    if ! python -c 'import torch, torch_npu; assert torch.npu.is_available()' >/dev/null 2>&1; then
        echo "[env_npu] torch_npu/NPU check failed (env=$CONDA_ENV, CANN=$CANN_HOME, NPU=$NPU_ID)" >&2
        return 1 2>/dev/null || exit 1
    fi
fi

echo "[env_npu] env=$CONDA_ENV cann=$CANN_HOME npu=$NPU_ID python=$(command -v python)"
