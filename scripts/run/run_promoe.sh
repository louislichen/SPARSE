#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
# shellcheck disable=SC1090
source "${ENV_FILE}"
METHOD="${1:-promoe-style}"
case "${METHOD}" in
  promoe-style) PLANNER=promoe; EXECUTOR_CONTROL=0 ;;
  rr-matched) PLANNER=none; EXECUTOR_CONTROL=1 ;;
  *) echo "Usage: run_promoe.sh [promoe-style|rr-matched]" >&2; exit 2 ;;
esac
DATASET_PATH="${DATASET_PATH_OVERRIDE:-${C4_RUNTIME:-}}"
for key in PYTHON_BIN MODEL_PATH DATASET_PATH RESULTS_ROOT; do
  [[ -n "${!key:-}" ]] || { echo "[promoe][ERROR] ${key} is required." >&2; exit 2; }
done
if [[ "${PLANNER}" == "promoe" && -z "${PROMOE_DIR:-}" ]]; then
  echo "[promoe][ERROR] PROMOE_DIR is required." >&2; exit 2
fi
# Explicitly supplied topology; no two-node/four-GPU constraint.
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NODE_RANK="${NODE_RANK:-0}"
for key in NNODES NPROC_PER_NODE; do
  [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || { echo "[promoe][ERROR] ${key} must be positive." >&2; exit 2; }
done
[[ "${NODE_RANK}" =~ ^[0-9]+$ && "${NODE_RANK}" -lt "${NNODES}" ]] || {
  echo "[promoe][ERROR] NODE_RANK must be in [0, NNODES)." >&2; exit 2;
}
WORLD_SIZE_CURRENT=$((NNODES * NPROC_PER_NODE))
EP_SIZE="${EP_SIZE:-${WORLD_SIZE_CURRENT}}"
if [[ "${EP_SIZE}" != "${WORLD_SIZE_CURRENT}" ]]; then
  echo "[promoe][ERROR] This launcher requires EP_SIZE=NNODES*NPROC_PER_NODE." >&2; exit 2
fi
MODEL_KEY="${MODEL_KEY:-model}"
NETWORK_LABEL="${NETWORK_LABEL:-physical}"
BATCH_CURRENT="${BATCH_OVERRIDE:-4}"
PROMPT_CURRENT="${PROMPT_LEN_OVERRIDE:-512}"
OUTPUT_CURRENT="${MAX_NEW_OVERRIDE:-128}"
REQUEST_GROUP="${REQUEST_GROUP:-0}"
[[ "${REQUEST_GROUP}" =~ ^[0-9]+$ ]] || { echo "[promoe][ERROR] REQUEST_GROUP must be non-negative." >&2; exit 2; }
DATASET_SKIP_CURRENT="${DATASET_SKIP_OVERRIDE:-$((REQUEST_GROUP * ${REQUEST_GROUP_STRIDE:-32}))}"
WORKLOAD="${WORKLOAD_TAG:-b${BATCH_CURRENT}-p${PROMPT_CURRENT}-o${OUTPUT_CURRENT}}"
GROUP_TAG="$(printf 'group%02d' "${REQUEST_GROUP}")"
OUT_DIR="${RESULTS_ROOT}/runtime/${RUN_ID:-promoe-development}/${MODEL_KEY}/${NETWORK_LABEL}/${WORKLOAD}/${GROUP_TAG}/${METHOD}"
export OUT_DIR METHOD NNODES NPROC_PER_NODE NODE_RANK EP_SIZE
export MODEL_KEY NETWORK_LABEL WORKLOAD REQUEST_GROUP BATCH_CURRENT PROMPT_CURRENT OUTPUT_CURRENT DATASET_SKIP_CURRENT
export WARMUP="${WARMUP:-1}" REPEAT="${REPEAT:-3}"
export PROMOE_DISTANCE="${PROMOE_DISTANCE:-2}"
export PROMOE_PREDICT_PREFILL="${PROMOE_PREDICT_PREFILL:-0}"
export EXPERT_CACHE="${EXPERT_CACHE:-8}"
export CACHE_BY_RANK="${CACHE_BY_RANK:-}"
mkdir -p "${OUT_DIR}/logs"
cd "${ROOT}"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  for path in "${PYTHON_BIN}" "${MODEL_PATH}" "${DATASET_PATH}"; do
    [[ -e "${path}" ]] || { echo "[promoe][ERROR] missing ${path}" >&2; exit 2; }
  done
  "${PYTHON_BIN}" - <<'PY'
import json, os
from pathlib import Path
keys = ("METHOD", "NNODES", "NPROC_PER_NODE", "NODE_RANK", "EP_SIZE",
        "EXPERT_CACHE", "CACHE_BY_RANK", "PROMOE_DISTANCE", "PROMOE_PREDICT_PREFILL")
config = {k.lower(): os.environ[k] for k in keys}
config.update(model=os.environ["MODEL_KEY"], network=os.environ["NETWORK_LABEL"],
              workload=os.environ["WORKLOAD"], request_group=int(os.environ["REQUEST_GROUP"]),
              batch=int(os.environ["BATCH_CURRENT"]), prompt_tokens=int(os.environ["PROMPT_CURRENT"]),
              output_tokens=int(os.environ["OUTPUT_CURRENT"]), dataset_skip=int(os.environ["DATASET_SKIP_CURRENT"]),
              warmup=int(os.environ["WARMUP"]), repeat=int(os.environ["REPEAT"]))
capacities = [int(c) for c in os.environ["CACHE_BY_RANK"].split(",")] if os.environ["CACHE_BY_RANK"] else []
if capacities and len(capacities) != int(os.environ["EP_SIZE"]):
    raise ValueError("CACHE_BY_RANK must contain one capacity per EP rank")
if os.environ["METHOD"] == "promoe-style" and any(c < 1 for c in (capacities or [int(os.environ["EXPERT_CACHE"])])):
    raise ValueError("ProMoE-style requires positive cache capacities")
Path(os.environ["OUT_DIR"], "case_config.json").write_text(
    json.dumps(config, indent=2) + "\n")
PY
fi
CMD=("${PYTHON_BIN}" -m torch.distributed.run
  --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" --node_rank="${NODE_RANK}"
  --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29671}"
  scripts/perf.py --runtime-mode baseline --ep-size "${EP_SIZE}" --model-path "${MODEL_PATH}"
  --dataset "${DATASET_PATH}" --dtype float16 --batch "${BATCH_CURRENT}"
  --prompt-len "${PROMPT_CURRENT}" --max-new "${OUTPUT_CURRENT}"
  --dataset-skip-valid "${DATASET_SKIP_CURRENT}" --max-scan-samples "${MAX_SCAN_SAMPLES:-2000}"
  --placement-mode round_robin --routing-mode router --expert-source router
  --expert-offload cpu --expert-cache "${EXPERT_CACHE}" --cache-policy lru
  --planner-mode "${PLANNER}" --repeat "${REPEAT:-3}" --warmup "${WARMUP:-1}"
  --out-dir "${OUT_DIR}" --run-tag "${METHOD}" --measure-ttft)
ENV_ARGS=(env "PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" "OMP_NUM_THREADS=${PROMOE_CPU_THREADS:-1}"
  "TOKENIZERS_PARALLELISM=false" "TORCHDYNAMO_DISABLE=1"
  "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}" "GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}"
  "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}"
  "SPARSE_PROMOE_DIR=${PROMOE_DIR:-}" "SPARSE_PROMOE_DISTANCE=${PROMOE_DISTANCE}"
  "SPARSE_PROMOE_PREDICT_PREFILL=${PROMOE_PREDICT_PREFILL}" "SPARSE_EXPERT_CACHE_BY_RANK=${CACHE_BY_RANK}"
  "SPARSE_PLAIN_EP_EXEC_MODE=sorted_segment" "SPARSE_EP_FUSED_WEIGHT_DISPATCH=0"
  "SPARSE_PROMOE_EXECUTOR_CONTROL=${EXECUTOR_CONTROL}"
  "SPARSE_EPIPE_ENABLE=0" "SPARSE_EPIPE_PREFILL_ENABLE=0" "SPARSE_RS_TWO_STAGE=0"
  "SPARSE_HOTPIN_ENABLE=0" "SPARSE_MOE_INFINITY_POLICY=0" "SPARSE_ERS_ENABLE=0"
  "SPARSE_NODEPATCH_ENABLE=0" "RUN_SKVS=0" "SPARSE_GLOBAL_EXPERT_CACHE=0"
  "SPARSE_SCP_APPLY_PREFILL=0" "SPARSE_SCP_APPLY_DECODE=0" "SPARSE_SCP_TRANSIENT_PREFILL_CACHE=0"
  "SPARSE_NODE_GROUPED_DISPATCH=0" "SPARSE_RANK_GROUPED_DISPATCH=0"
  "SPARSE_SIM_ENABLE_WEAKNET=0" "SPARSE_LOGICAL_WEAKNET_ENABLE=0" "SPARSE_SIM_ENABLE_HETEROGENEITY=0"
  "SPARSE_PREFILL_LAST_LOGITS_ONLY=1" "SPARSE_RESET_RUNTIME_STATE_EACH_REPEAT=1"
  "SPARSE_CLEAR_EXPERT_CACHE_EACH_REPEAT=1")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[DRY_RUN]'; printf ' %q' "${ENV_ARGS[@]}" "${CMD[@]}"; printf '\n'; exit 0
fi
"${ENV_ARGS[@]}" "${CMD[@]}" 2>&1 | tee "${OUT_DIR}/logs/node${NODE_RANK}.log"
