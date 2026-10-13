#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
# shellcheck disable=SC1090
source "${ENV_FILE}"
for key in PYTHON_BIN MODEL_PATH C4_PROFILE_TRAIN PROMOE_PROFILE_DIR; do
  [[ -n "${!key:-}" ]] || { echo "[profile-promoe][ERROR] ${key} is required." >&2; exit 2; }
done
cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${PROMOE_PROFILE_CUDA_VISIBLE_DEVICES:-0}" \
"${PYTHON_BIN}" scripts/profile_next_layer_rs.py \
  --model-path "${MODEL_PATH}" --dataset "${C4_PROFILE_TRAIN}" --out-dir "${PROMOE_PROFILE_DIR}" \
  --feature-source decoder_input --num-samples "${PROMOE_NUM_SAMPLES:-4096}" \
  --prompt-len "${PROMOE_PROFILE_PROMPT_LEN:-512}" --decode-steps "${PROMOE_DECODE_STEPS:-64}" \
  --decode-token-source "${PROMOE_DECODE_TOKEN_SOURCE:-dataset}" \
  --profile-batch-size "${PROMOE_PROFILE_BATCH_SIZE:-1}" \
  --max-scan-samples "${PROMOE_MAX_SCAN_SAMPLES:-20000}" --device-map auto \
  --max-memory-per-gpu "${PROMOE_MAX_MEMORY_PER_GPU:-9GiB}" \
  --max-memory-cpu "${PROMOE_MAX_MEMORY_CPU:-80GiB}" \
  --attn-implementation eager --local-files-only --seed 42
