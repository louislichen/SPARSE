#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
# shellcheck disable=SC1090
source "${ENV_FILE}"

for key in PYTHON_BIN RS_PROFILE_DIR RS_DIR; do
  [[ -n "${!key:-}" ]] || { echo "[train-rs][ERROR] ${key} is required." >&2; exit 2; }
done
[[ -s "${RS_PROFILE_DIR}/manifest.json" ]] || { echo "[train-rs][ERROR] missing RS profile manifest." >&2; exit 2; }
mkdir -p "${RS_DIR}"

cd "${ROOT}"
"${PYTHON_BIN}" scripts/validate_next_layer_rs_profile.py \
  --profile-base-dir "${RS_PROFILE_DIR}" \
  --check-load

CUDA_VISIBLE_DEVICES="${RS_TRAIN_CUDA_VISIBLE_DEVICES:-0,1}" \
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${RS_TRAIN_NPROC:-2}" \
  scripts/train_next_layer_rs.py \
  --profile-base-dir "${RS_PROFILE_DIR}" \
  --output-dir "${RS_DIR}" \
  --lead-layers 1 \
  --model-arch "${RS_MODEL_ARCH:-residual}" \
  --hidden-dim "${RS_HIDDEN_DIM:-512}" \
  --num-hidden-layers "${RS_NUM_HIDDEN_LAYERS:-2}" \
  --dropout 0.05 \
  --batch-size "${RS_BATCH_SIZE:-512}" \
  --epochs "${RS_EPOCHS:-40}" \
  --lr 8e-4 \
  --weight-decay 1e-4 \
  --early-stop-patience 5 \
  --eval-ks 1,2,4,8,12 \
  --feature-mode hidden \
  --target-mode uniform_multihot \
  --loss-mode ce \
  --seed 42

checkpoint_count="$(find "${RS_DIR}" -maxdepth 1 -name 'rs_l*_inL*.pth' -type f | wc -l | tr -d ' ')"
if [[ "${checkpoint_count}" == "0" ]]; then
  echo "[train-rs][ERROR] training produced no per-layer checkpoints." >&2
  exit 2
fi
echo "[train-rs] checkpoints=${checkpoint_count} dir=${RS_DIR}"
