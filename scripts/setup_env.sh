#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-sparse-aaai}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[setup][ERROR] conda is required." >&2
  exit 2
fi

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install --index-url "${PYTORCH_INDEX_URL}" \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
python -m pip install -r "${ROOT}/requirements.txt"

echo "[setup] environment ready: ${ENV_NAME}"
echo "[setup] python: $(command -v python)"
