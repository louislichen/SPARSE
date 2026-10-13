#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
# shellcheck disable=SC1090
source "${ENV_FILE}"

for key in REMOTE_SSH REMOTE_REPO REMOTE_ENV_FILE RESULTS_ROOT; do
  if [[ -z "${!key:-}" ]]; then
    echo "[launch][ERROR] ${key} is required in ${ENV_FILE}" >&2
    exit 2
  fi
done
for value in "${REMOTE_REPO}" "${REMOTE_ENV_FILE}"; do
  if [[ "${value}" == *"'"* ]]; then
    echo "[launch][ERROR] paths containing single quotes are unsupported." >&2
    exit 2
  fi
done

RUN_ID="${RUN_ID:-paper-main-$(date '+%Y%m%d-%H%M%S')}"
SESSION="${SESSION:-sparse-${RUN_ID}}"
RUN_COMMAND="${RUN_COMMAND:-bash scripts/run/run_matrix.sh}"
METHODS_VALUE="${METHODS:-sparse}"
WORKLOADS_VALUE="${WORKLOADS:-b1 b2 b4}"
REQUEST_GROUPS_VALUE="${REQUEST_GROUPS:-0 1 2 3}"
for value in "${METHODS_VALUE}" "${WORKLOADS_VALUE}" "${REQUEST_GROUPS_VALUE}" "${RUN_COMMAND}"; do
  if [[ "${value}" == *"'"* ]]; then
    echo "[launch][ERROR] launch values containing single quotes are unsupported." >&2
    exit 2
  fi
done
RUN_ENV="RUN_ID='${RUN_ID}' METHODS='${METHODS_VALUE}' WORKLOADS='${WORKLOADS_VALUE}' REQUEST_GROUPS='${REQUEST_GROUPS_VALUE}' WARMUP='${WARMUP:-1}' REPEAT='${REPEAT:-3}'"
REMOTE_COMMAND="cd '${REMOTE_REPO}' && ${RUN_ENV} ENV_FILE='${REMOTE_ENV_FILE}' ${RUN_COMMAND}"
LOCAL_COMMAND="cd '${ROOT}' && ${RUN_ENV} ENV_FILE='${ENV_FILE}' ${RUN_COMMAND}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] ssh ${REMOTE_SSH} tmux new-session -d -s ${SESSION} ${REMOTE_COMMAND}"
  echo "[DRY_RUN] tmux new-session -d -s ${SESSION} ${LOCAL_COMMAND}"
  exit 0
fi

command -v tmux >/dev/null || { echo "[launch][ERROR] tmux is required locally." >&2; exit 2; }
ssh "${REMOTE_SSH}" "command -v tmux >/dev/null"
ssh "${REMOTE_SSH}" "tmux has-session -t '${SESSION}' 2>/dev/null && tmux kill-session -t '${SESSION}' || true"
tmux has-session -t "${SESSION}" 2>/dev/null && tmux kill-session -t "${SESSION}" || true

ssh "${REMOTE_SSH}" "tmux new-session -d -s '${SESSION}' \"${REMOTE_COMMAND}\""
sleep 2
tmux new-session -d -s "${SESSION}" "${LOCAL_COMMAND}"

echo "[launch] started run_id=${RUN_ID}"
echo "[launch] local:  tmux attach -t ${SESSION}"
echo "[launch] remote: ssh ${REMOTE_SSH} -t 'tmux attach -t ${SESSION}'"
