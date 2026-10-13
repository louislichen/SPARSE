#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
# shellcheck disable=SC1090
source "${ENV_FILE}"
for key in PYTHON_BIN PROMOE_PROFILE_DIR PROMOE_DIR; do
  [[ -n "${!key:-}" ]] || { echo "[train-promoe][ERROR] ${key} is required." >&2; exit 2; }
done
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" scripts/validate_next_layer_rs_profile.py \
  --profile-base-dir "${PROMOE_PROFILE_DIR}" --check-load
# This is an independent two-linear-layer predictor trained at a different
# distance; never pass SPARSE RS checkpoints as a silent fallback.
CUDA_VISIBLE_DEVICES="${PROMOE_TRAIN_CUDA_VISIBLE_DEVICES:-0}" \
"${PYTHON_BIN}" -m torch.distributed.run --standalone \
  --nproc_per_node="${PROMOE_TRAIN_NPROC:-1}" scripts/train_next_layer_rs.py \
  --profile-base-dir "${PROMOE_PROFILE_DIR}" --output-dir "${PROMOE_DIR}" \
  --lead-layers "${PROMOE_DISTANCE:-2}" --model-arch promoe \
  --hidden-dim "${PROMOE_HIDDEN_DIM:-1024}" --feature-mode hidden \
  --target-mode uniform_multihot --loss-mode ce --batch-size "${PROMOE_BATCH_SIZE:-512}" \
  --epochs "${PROMOE_EPOCHS:-40}" --lr "${PROMOE_LR:-8e-4}" --weight-decay 1e-4 \
  --early-stop-patience 5 --seed 42
