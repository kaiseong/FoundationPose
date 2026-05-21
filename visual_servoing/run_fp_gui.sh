#!/usr/bin/env bash
set -euo pipefail

CONDA_BASE="${CONDA_BASE:-/home/kgs/miniforge3}"
ENV_NAME="${ENV_NAME:-visual}"
DATA_ROOT="${DATA_ROOT:-/home/kgs/Hierarchical_lerobot/visual_servoing_data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FOUNDATIONPOSE_ROOT="${FOUNDATIONPOSE_ROOT:-${PROJECT_ROOT}}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export FOUNDATIONPOSE_ROOT
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

python -m visual_servoing.scripts.fp_gui \
  --foundationpose-root "${FOUNDATIONPOSE_ROOT}" \
  --data-root "${DATA_ROOT}" \
  "$@"
