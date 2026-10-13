#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[run-matrix][ERROR] missing ${ENV_FILE}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

RUN_ID="${RUN_ID:-paper-main-$(date '+%Y%m%d-%H%M%S')}"
METHODS="${METHODS:-sparse}"
WORKLOADS="${WORKLOADS:-b1 b2 b4}"
REQUEST_GROUPS="${REQUEST_GROUPS:-0 1 2 3}"
export RUN_ID ENV_FILE

QUEUE_DIR="${RESULTS_ROOT}/runtime/${RUN_ID}/queue"
mkdir -p "${QUEUE_DIR}"
exec > >(tee -a "${QUEUE_DIR}/node${NODE_RANK}.log") 2>&1

echo "[run-matrix] run_id=${RUN_ID} node_rank=${NODE_RANK}"
echo "[run-matrix] methods=${METHODS} workloads=${WORKLOADS} request_groups=${REQUEST_GROUPS}"

for group in ${REQUEST_GROUPS}; do
  for workload in ${WORKLOADS}; do
    for method in ${METHODS}; do
      bash "${ROOT}/scripts/run/run_case.sh" "${method}" "${workload}" "${group}"
    done
  done
done

date -u '+%Y-%m-%dT%H:%M:%SZ' >"${QUEUE_DIR}/node${NODE_RANK}.complete"
echo "[run-matrix] COMPLETE node_rank=${NODE_RANK}"
