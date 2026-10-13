#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
# shellcheck disable=SC1090
source "${ENV_FILE}"

for key in PYTHON_BIN MODEL_PATH C4_PROFILE_TRAIN RS_PROFILE_DIR; do
  [[ -n "${!key:-}" ]] || { echo "[profile-rs][ERROR] ${key} is required." >&2; exit 2; }
done

mkdir -p "${RS_PROFILE_DIR}"
cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${RS_PROFILE_CUDA_VISIBLE_DEVICES:-0,1}" \
"${PYTHON_BIN}" scripts/profile_next_layer_rs.py \
  --model-path "${MODEL_PATH}" \
  --dataset "${C4_PROFILE_TRAIN}" \
  --out-dir "${RS_PROFILE_DIR}" \
  --num-samples "${RS_NUM_SAMPLES:-4096}" \
  --prompt-len 512 \
  --decode-steps "${RS_DECODE_STEPS:-64}" \
  --decode-token-source "${RS_DECODE_TOKEN_SOURCE:-dataset}" \
  --profile-batch-size "${RS_PROFILE_BATCH_SIZE:-1}" \
  --max-scan-samples "${RS_MAX_SCAN_SAMPLES:-20000}" \
  --save-interval-records "${RS_SAVE_INTERVAL_RECORDS:-4096}" \
  --device-map auto \
  --max-memory-per-gpu "${RS_MAX_MEMORY_PER_GPU:-9GiB}" \
  --max-memory-cpu "${RS_MAX_MEMORY_CPU:-80GiB}" \
  --attn-implementation eager \
  --local-files-only \
  --seed 42
