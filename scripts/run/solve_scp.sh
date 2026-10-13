#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
# shellcheck disable=SC1090
source "${ENV_FILE}"

[[ "${NODE_RANK:-0}" == "0" ]] || { echo "[solve-scp] node rank 1 has no solver work."; exit 0; }
for key in PYTHON_BIN SCP_TRAIN_PROFILE SCP_VALID_PROFILE TOPOLOGY_PATH SCP_PATH MODEL_NUM_EXPERTS; do
  [[ -n "${!key:-}" ]] || { echo "[solve-scp][ERROR] ${key} is required." >&2; exit 2; }
done
for path in "${SCP_TRAIN_PROFILE}" "${SCP_VALID_PROFILE}" "${TOPOLOGY_PATH}"; do
  [[ -s "${path}" ]] || { echo "[solve-scp][ERROR] missing input: ${path}" >&2; exit 2; }
done

mkdir -p "$(dirname "${SCP_PATH}")"
cd "${ROOT}"
"${PYTHON_BIN}" scripts/solve_replicated_scp_ep.py \
  --train-profile "${SCP_TRAIN_PROFILE}" \
  --valid-profile "${SCP_VALID_PROFILE}" \
  --topology "${TOPOLOGY_PATH}" \
  --out "${SCP_PATH}" \
  --stage prefill \
  --objective tail_headroom_scp_v2 \
  --global-extra-replicas 96 \
  --extra-replicas-per-rank-per-layer 2 \
  --max-replicas-per-expert 2 \
  --candidate-prefilter-topn 256 \
  --expert-cache-capacity "${EXPERT_CACHE:-8}" \
  --expert-cache-capacity-by-rank "${CACHE_BY_RANK:-8,8,8,8}" \
  --require-valid-positive \
  --require-inter-node-rankmax-bytes-gain \
  --min-tail-headroom-pct 0.5 \
  --num-experts "${MODEL_NUM_EXPERTS}"

"${PYTHON_BIN}" -m json.tool "${SCP_PATH}" >/dev/null
echo "[solve-scp] placement ready: ${SCP_PATH}"
