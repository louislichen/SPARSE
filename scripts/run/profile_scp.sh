#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
# shellcheck disable=SC1090
source "${ENV_FILE}"

STAGE="${1:-all}"
for key in PYTHON_BIN NODE_RANK SCP_PROFILE_ROOT C4_PROFILE_TRAIN C4_PROFILE_VALID; do
  [[ -n "${!key:-}" ]] || { echo "[profile-scp][ERROR] ${key} is required." >&2; exit 2; }
done

collect_split() {
  local split="$1"
  local dataset="$2"
  local shards="$3"
  local skip_base="$4"
  local shard skip shard_dir
  for ((shard=0; shard<shards; shard++)); do
    skip=$((skip_base + shard * ${SCP_PROFILE_SHARD_STRIDE:-4}))
    shard_dir="${SCP_PROFILE_ROOT}/${split}_shards/shard_${shard}"
    mkdir -p "${shard_dir}"
    export RUN_ID="scp-profile-${MODEL_KEY}-${split}-$(printf '%04d' "${shard}")"
    export DATASET_PATH_OVERRIDE="${dataset}"
    export DATASET_SKIP_OVERRIDE="${skip}"
    export BATCH_OVERRIDE="${SCP_PROFILE_BATCH:-2}"
    export PROMPT_LEN_OVERRIDE=512
    export MAX_NEW_OVERRIDE=1
    export WARMUP=0
    export REPEAT=1
    export SPARSE_PROFILE_ROUTING=1
    export SPARSE_PROFILE_ROUTING_OUT="${shard_dir}"
    export SPARSE_PROFILE_ROUTING_AGG=1
    export SPARSE_PROFILE_STAGE=prefill
    export SPARSE_PROFILE_ROUTING_SHARD_ID="${shard}"
    export SPARSE_PROFILE_ROUTING_REQUEST_ID="${skip}"
    export SPARSE_PROFILE_SPLIT="${split}"
    bash "${ROOT}/scripts/run/run_case.sh" rr b2 0
  done
}

gather_remote() {
  [[ "${NODE_RANK}" == "0" ]] || return 0
  local remote_root="${REMOTE_SCP_PROFILE_ROOT:-${REMOTE_RESULTS_ROOT:-}/offline/scp/${MODEL_KEY}}"
  [[ -n "${REMOTE_SSH:-}" && -n "${remote_root}" ]] || {
    echo "[profile-scp][ERROR] node 0 needs REMOTE_SSH and REMOTE_SCP_PROFILE_ROOT to gather rank files." >&2
    exit 2
  }
  mkdir -p "${SCP_PROFILE_ROOT}/train_shards" "${SCP_PROFILE_ROOT}/valid_shards"
  rsync -a "${REMOTE_SSH}:${remote_root}/train_shards/" "${SCP_PROFILE_ROOT}/train_shards/"
  rsync -a "${REMOTE_SSH}:${remote_root}/valid_shards/" "${SCP_PROFILE_ROOT}/valid_shards/"
}

merge_profiles() {
  [[ "${NODE_RANK}" == "0" ]] || return 0
  mkdir -p "${SCP_PROFILE_ROOT}/grouped"
  "${PYTHON_BIN}" "${ROOT}/scripts/scp_ep_merge_profiles.py" \
    --input-dir "${SCP_PROFILE_ROOT}/train_shards" \
    --out "${SCP_PROFILE_ROOT}/grouped/profile_train_grouped.npz"
  "${PYTHON_BIN}" "${ROOT}/scripts/scp_ep_merge_profiles.py" \
    --input-dir "${SCP_PROFILE_ROOT}/valid_shards" \
    --out "${SCP_PROFILE_ROOT}/grouped/profile_valid_grouped.npz"
}

case "${STAGE}" in
  train)
    collect_split train "${C4_PROFILE_TRAIN}" "${SCP_PROFILE_TRAIN_SHARDS:-256}" 0
    ;;
  valid)
    collect_split valid "${C4_PROFILE_VALID}" "${SCP_PROFILE_VALID_SHARDS:-64}" "${SCP_PROFILE_VALID_SKIP_BASE:-0}"
    ;;
  gather)
    gather_remote
    ;;
  merge)
    merge_profiles
    ;;
  all)
    collect_split train "${C4_PROFILE_TRAIN}" "${SCP_PROFILE_TRAIN_SHARDS:-256}" 0
    collect_split valid "${C4_PROFILE_VALID}" "${SCP_PROFILE_VALID_SHARDS:-64}" "${SCP_PROFILE_VALID_SKIP_BASE:-0}"
    gather_remote
    merge_profiles
    ;;
  *) echo "Usage: profile_scp.sh train|valid|gather|merge|all" >&2; exit 2 ;;
esac
