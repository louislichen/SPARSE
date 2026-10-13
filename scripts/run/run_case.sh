#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[run-case][ERROR] missing environment file: ${ENV_FILE}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

METHOD="${1:-}"
WORKLOAD="${2:-b4}"
REQUEST_GROUP="${3:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage: run_case.sh METHOD WORKLOAD REQUEST_GROUP

Methods:
  rr, lod, eplb, prism, static-hot, moe-infinity
  scp, rs, epipe, scp-rs, rs-epipe, sparse
  executor-no-epipe, executor-with-epipe

Workloads:
  b1 = batch 1, prompt 1024, output 32
  b2 = batch 2, prompt 512, output 64
  b4 = batch 4, prompt 512, output 128
EOF
}

if [[ -z "${METHOD}" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "${REQUEST_GROUP}" =~ ^[0-9]+$ ]]; then
  echo "[run-case][ERROR] REQUEST_GROUP must be non-negative." >&2
  exit 2
fi

case "${WORKLOAD}" in
  b1) BATCH=1; PROMPT_LEN=1024; MAX_NEW=32 ;;
  b2) BATCH=2; PROMPT_LEN=512; MAX_NEW=64 ;;
  b4) BATCH=4; PROMPT_LEN=512; MAX_NEW=128 ;;
  *) echo "[run-case][ERROR] unknown workload: ${WORKLOAD}" >&2; usage >&2; exit 2 ;;
esac
BATCH="${BATCH_OVERRIDE:-${BATCH}}"
PROMPT_LEN="${PROMPT_LEN_OVERRIDE:-${PROMPT_LEN}}"
MAX_NEW="${MAX_NEW_OVERRIDE:-${MAX_NEW}}"

required=(PYTHON_BIN CUDA_VISIBLE_DEVICES NODE_RANK NNODES NPROC_PER_NODE EP_SIZE MASTER_ADDR MODEL_PATH C4_RUNTIME RESULTS_ROOT)
for key in "${required[@]}"; do
  if [[ -z "${!key:-}" ]]; then
    echo "[run-case][ERROR] ${key} is not set in ${ENV_FILE}" >&2
    exit 2
  fi
done
if [[ "${NNODES}" != "2" || "${NPROC_PER_NODE}" != "2" || "${EP_SIZE}" != "4" ]]; then
  echo "[run-case][ERROR] formal protocol requires NNODES=2, NPROC_PER_NODE=2, EP_SIZE=4." >&2
  exit 2
fi

DATASET_PATH="${DATASET_PATH_OVERRIDE:-${C4_RUNTIME}}"
DATASET_SKIP="${DATASET_SKIP_OVERRIDE:-$((REQUEST_GROUP * ${REQUEST_GROUP_STRIDE:-32}))}"
RUN_ID="${RUN_ID:-sparse-run-$(date '+%Y%m%d-%H%M%S')}"
NETWORK_LABEL="${NETWORK_LABEL:-physical}"
EXPERT_CACHE_CURRENT="${EXPERT_CACHE:-8}"
CACHE_BY_RANK_CURRENT="${CACHE_BY_RANK:-8,8,8,8}"
CACHE_POLICY_CURRENT=lru
PLACEMENT_MODE=round_robin
PLACEMENT_PATH="${SCP_PATH:-}"
PLANNER_MODE=none
RUN_RS=0
SCP_PREFILL=0
EPIPE_ENABLE=0
EPIPE_RS_MODE=off
PLAIN_EXEC=unique_mask
HOTPIN_ENABLE=0
MOE_INFINITY_ENABLE=0

case "${METHOD}" in
  rr) ;;
  lod)
    EXPERT_CACHE_CURRENT=0
    CACHE_BY_RANK_CURRENT=0,0,0,0
    ;;
  eplb)
    PLACEMENT_MODE=scp_ep
    PLACEMENT_PATH="${EPLB_PATH:-}"
    SCP_PREFILL=1
    ;;
  prism)
    PLACEMENT_MODE=scp_ep
    PLACEMENT_PATH="${PRISM_PATH:-}"
    SCP_PREFILL=1
    ;;
  static-hot)
    PLACEMENT_MODE=scp_ep
    SCP_PREFILL=1
    CACHE_POLICY_CURRENT=hotpin_lru
    HOTPIN_ENABLE=1
    ;;
  moe-infinity)
    PLACEMENT_MODE=scp_ep
    SCP_PREFILL=1
    MOE_INFINITY_ENABLE=1
    ;;
  scp)
    PLACEMENT_MODE=scp_ep
    SCP_PREFILL=1
    ;;
  rs)
    RUN_RS=1
    PLANNER_MODE=rs_prefetch
    ;;
  epipe)
    EPIPE_ENABLE=1
    PLAIN_EXEC=sorted_segment
    ;;
  scp-rs)
    PLACEMENT_MODE=scp_ep
    SCP_PREFILL=1
    RUN_RS=1
    PLANNER_MODE=rs_prefetch
    ;;
  rs-epipe)
    RUN_RS=1
    PLANNER_MODE=rs_prefetch
    EPIPE_ENABLE=1
    EPIPE_RS_MODE=residual_scan
    PLAIN_EXEC=sorted_segment
    ;;
  sparse)
    PLACEMENT_MODE=scp_ep
    SCP_PREFILL=1
    RUN_RS=1
    PLANNER_MODE=rs_prefetch
    EPIPE_ENABLE=1
    EPIPE_RS_MODE=residual_scan
    PLAIN_EXEC=sorted_segment
    ;;
  executor-no-epipe)
    PLACEMENT_MODE=scp_ep
    SCP_PREFILL=1
    RUN_RS=1
    PLANNER_MODE=rs_prefetch
    PLAIN_EXEC=sorted_segment
    ;;
  executor-with-epipe)
    PLACEMENT_MODE=scp_ep
    SCP_PREFILL=1
    RUN_RS=1
    PLANNER_MODE=rs_prefetch
    EPIPE_ENABLE=1
    EPIPE_RS_MODE=residual_scan
    PLAIN_EXEC=sorted_segment
    ;;
  *) echo "[run-case][ERROR] unknown method: ${METHOD}" >&2; usage >&2; exit 2 ;;
esac

needs_scp=0
[[ "${PLACEMENT_MODE}" == "scp_ep" ]] && needs_scp=1
if [[ "${DRY_RUN}" != "1" ]]; then
  for path in "${PYTHON_BIN}" "${MODEL_PATH}" "${DATASET_PATH}"; do
    [[ -e "${path}" ]] || { echo "[run-case][ERROR] missing path: ${path}" >&2; exit 2; }
  done
  if [[ "${needs_scp}" == "1" && ! -s "${PLACEMENT_PATH}" ]]; then
    echo "[run-case][ERROR] missing placement for ${METHOD}: ${PLACEMENT_PATH}" >&2
    exit 2
  fi
  if [[ "${RUN_RS}" == "1" && ! -d "${RS_DIR:-}" ]]; then
    echo "[run-case][ERROR] missing RS checkpoint directory: ${RS_DIR:-}" >&2
    exit 2
  fi
  if [[ "${HOTPIN_ENABLE}" == "1" && ! -s "${HOT_PROFILE_PATH:-}" ]]; then
    echo "[run-case][ERROR] missing HOT_PROFILE_PATH" >&2
    exit 2
  fi
  if [[ "${MOE_INFINITY_ENABLE}" == "1" && ! -s "${MOE_INFINITY_EAMC_PATH:-}" ]]; then
    echo "[run-case][ERROR] missing MOE_INFINITY_EAMC_PATH" >&2
    exit 2
  fi
fi

GROUP_TAG="$(printf 'group%02d' "${REQUEST_GROUP}")"
OUT_DIR="${RESULTS_ROOT}/runtime/${RUN_ID}/${MODEL_KEY:-model}/${NETWORK_LABEL}/${WORKLOAD}/${GROUP_TAG}/${METHOD}"
mkdir -p "${OUT_DIR}/logs"
cat >"${OUT_DIR}/case_config.json" <<EOF
{
  "method": "${METHOD}",
  "model": "${MODEL_KEY:-model}",
  "network": "${NETWORK_LABEL}",
  "workload": "${WORKLOAD}",
  "batch": ${BATCH},
  "prompt_tokens": ${PROMPT_LEN},
  "output_tokens": ${MAX_NEW},
  "request_group": ${REQUEST_GROUP},
  "dataset_skip": ${DATASET_SKIP},
  "warmup": ${WARMUP:-1},
  "repeat": ${REPEAT:-3},
  "cache_by_rank": "${CACHE_BY_RANK_CURRENT}",
  "prefetch_top_m": ${PREFETCH_TOP_M:-2},
  "prediction_top_k": ${PREDICTION_TOP_K:-4}
}
EOF

LAUNCHER=("${PYTHON_BIN}" -m torch.distributed.run
  --nproc_per_node="${NPROC_PER_NODE}"
  --nnodes="${NNODES}"
  --node_rank="${NODE_RANK}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT:-29671}")

CMD=("${LAUNCHER[@]}" scripts/perf.py
  --runtime-mode baseline
  --ep-size "${EP_SIZE}"
  --model-path "${MODEL_PATH}"
  --dataset "${DATASET_PATH}"
  --scp-path "${PLACEMENT_PATH}"
  --rs-dir "${RS_DIR:-}"
  --dtype float16
  --batch "${BATCH}"
  --prompt-len "${PROMPT_LEN}"
  --dataset-skip-valid "${DATASET_SKIP}"
  --max-scan-samples "${MAX_SCAN_SAMPLES:-2000}"
  --max-new "${MAX_NEW}"
  --placement-mode "${PLACEMENT_MODE}"
  --routing-mode router
  --expert-source router
  --expert-offload cpu
  --expert-cache "${EXPERT_CACHE_CURRENT}"
  --cache-policy "${CACHE_POLICY_CURRENT}"
  --planner-mode "${PLANNER_MODE}"
  --prefetch-top-m "${PREFETCH_TOP_M:-2}"
  --repeat "${REPEAT:-3}"
  --warmup "${WARMUP:-1}"
  --out-dir "${OUT_DIR}"
  --run-tag "${METHOD}"
  --measure-ttft)

ENV_ARGS=(env
  "PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}"
  "GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}"
  "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}"
  "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
  "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
  "TOKENIZERS_PARALLELISM=false"
  "OMP_NUM_THREADS=1"
  "TORCHDYNAMO_DISABLE=1"
  "SPARSE_EXPERT_CACHE_BY_RANK=${CACHE_BY_RANK_CURRENT}"
  "SPARSE_SCP_EP_PLACEMENT=${PLACEMENT_PATH}"
  "SPARSE_SCP_APPLY_PREFILL=${SCP_PREFILL}"
  "SPARSE_SCP_APPLY_DECODE=0"
  "SPARSE_SCP_TRANSIENT_PREFILL_CACHE=1"
  "SPARSE_SCP_CLEAR_CACHE_BEFORE_DECODE=0"
  "SPARSE_RS_PREFETCH_POLICY=cache_safe"
  "SPARSE_RS_PREFETCH_PRED_TOPK=${PREDICTION_TOP_K:-4}"
  "SPARSE_RS_PREFETCH_TOP_M_BY_RANK=2,2,2,2"
  "SPARSE_RS_PREFETCH_PRED_TOPK_BY_RANK=4,4,4,4"
  "SPARSE_RS_CACHE_SAFE_MIN_SCORE_BY_RANK=0.04,0.04,0.04,0.04"
  "SPARSE_RS_CACHE_SAFE_REPLACE_ADVANTAGE_BY_RANK=0.08,0.08,0.08,0.08"
  "SPARSE_RS_CACHE_SAFE_REPLACE_MARGIN_BY_RANK=1.20,1.20,1.20,1.20"
  "SPARSE_RS_PREFETCH_RECENT_PROTECT_STEPS_BY_RANK=1,1,1,1"
  "SPARSE_PREFETCH_MIN_CAND_SCORE=0.04"
  "SPARSE_PREFETCH_REPLACE_ADVANTAGE=0.08"
  "SPARSE_PREFETCH_REPLACE_MARGIN=1.20"
  "SPARSE_PREFETCH_ALLOW_ACTIVE_REPLACE=${RUN_RS}"
  "SPARSE_RS_PREFETCH_ALLOW_ACTIVE_REPLACE=${RUN_RS}"
  "SPARSE_EPIPE_ENABLE=${EPIPE_ENABLE}"
  "SPARSE_EPIPE_DECODE_ONLY=1"
  "SPARSE_EPIPE_PREFILL_ENABLE=0"
  "SPARSE_EPIPE_MODE=upfront_scan"
  "SPARSE_EPIPE_DECODE_RS_MODE=${EPIPE_RS_MODE}"
  "SPARSE_EPIPE_DECODE_WITH_RS_MODE=${EPIPE_RS_MODE}"
  "SPARSE_EPIPE_RS_HANDLE_LATE_PREFETCH=${EPIPE_ENABLE}"
  "SPARSE_EPIPE_RS_HANDLE_DEMAND_MISS=${EPIPE_ENABLE}"
  "SPARSE_EPIPE_RS_FORCE_DEMAND_STAGE=${EPIPE_ENABLE}"
  "SPARSE_EPIPE_NO_EVICT=1"
  "SPARSE_EPIPE_TRANSIENT_STAGING=1"
  "SPARSE_EPIPE_MAX_STAGED_EXPERTS=1"
  "SPARSE_EPIPE_MIN_FREE_MB=384"
  "SPARSE_PLAIN_EP_EXEC_MODE=${PLAIN_EXEC}"
  "SPARSE_EP_FUSED_WEIGHT_DISPATCH=0"
  "SPARSE_HOTPIN_ENABLE=${HOTPIN_ENABLE}"
  "SPARSE_HOTPIN_PROFILE_PATH=${HOT_PROFILE_PATH:-}"
  "SPARSE_HOTPIN_TOPK_PER_LAYER=2"
  "SPARSE_MOE_INFINITY_POLICY=${MOE_INFINITY_ENABLE}"
  "SPARSE_MOE_INFINITY_EAMC_PATH=${MOE_INFINITY_EAMC_PATH:-}"
  "SPARSE_MOE_INFINITY_WINDOW_LAYERS=64"
  "SPARSE_MOE_INFINITY_TOPK_PER_LAYER=${PREFETCH_TOP_M:-2}"
  "SPARSE_RESET_RUNTIME_STATE_EACH_REPEAT=1"
  "SPARSE_CLEAR_EXPERT_CACHE_EACH_REPEAT=1"
  "SPARSE_DEMOTE_HOT_CACHE_EACH_REPEAT=0"
  "SPARSE_PREFILL_LAST_LOGITS_ONLY=1"
  "SPARSE_DEMAND_H2D_MIN_FREE_MB=32"
  "SPARSE_PREFETCH_MIN_FREE_MB=32"
  "SPARSE_ERS_ENABLE=0"
  "SPARSE_NODEPATCH_ENABLE=0"
  "RUN_SKVS=0"
  "SPARSE_SIM_ENABLE_WEAKNET=0"
  "SPARSE_LOGICAL_WEAKNET_ENABLE=0"
  "SPARSE_SIM_ENABLE_HETEROGENEITY=0"
  "SPARSE_NODE_GROUPED_DISPATCH=0"
  "SPARSE_NODE_GROUPED_PREFILL=0"
  "SPARSE_RANK_GROUPED_DISPATCH=0"
  "SPARSE_RANK_GROUPED_PREFILL=0")

printf '[run-case] node=%s method=%s workload=%s request_group=%s out=%s\n' \
  "${NODE_RANK}" "${METHOD}" "${WORKLOAD}" "${REQUEST_GROUP}" "${OUT_DIR}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[DRY_RUN]'
  printf ' %q' "${ENV_ARGS[@]}" "${CMD[@]}"
  printf '\n'
  exit 0
fi

cd "${ROOT}"
"${ENV_ARGS[@]}" "${CMD[@]}" 2>&1 | tee "${OUT_DIR}/logs/${METHOD}.log"
