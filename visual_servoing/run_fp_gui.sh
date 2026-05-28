#!/usr/bin/env bash
set -euo pipefail

CONDA_BASE="${CONDA_BASE:-${HOME}/miniforge3}"
ENV_NAME="${ENV_NAME:-visual}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/visual_servoing_data}"
DEFAULT_FOUNDATIONPOSE_ROOT="${HOME}/FoundationPose"
if [[ -d "${DEFAULT_FOUNDATIONPOSE_ROOT}" ]]; then
  FOUNDATIONPOSE_ROOT="${FOUNDATIONPOSE_ROOT:-${DEFAULT_FOUNDATIONPOSE_ROOT}}"
else
  FOUNDATIONPOSE_ROOT="${FOUNDATIONPOSE_ROOT:-${PROJECT_ROOT}}"
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export FOUNDATIONPOSE_ROOT
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TORCH_LIB_DIR="$("${CONDA_PREFIX}/bin/python" - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

python -m visual_servoing.scripts.fp_gui \
  --foundationpose-root "${FOUNDATIONPOSE_ROOT}" \
  --data-root "${DATA_ROOT}" \
  "$@"
