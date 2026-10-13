#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-help}"

case "${ACTION}" in
  validate)
    bash "${ROOT}/scripts/validate_release.sh"
    ;;
  check-env)
    "${PYTHON_BIN:-python3}" "${ROOT}/scripts/check_env.py" --env-file "${ENV_FILE:-${ROOT}/.env}" --strict
    ;;
  reference-figures)
    "${PYTHON_BIN:-python3}" "${ROOT}/scripts/figures/plot_reference_results.py" \
      --input "${ROOT}/results/reference/source_policy_normalized.csv" \
      --out-dir "${ROOT}/results/figures"
    ;;
  profile-rs)
    bash "${ROOT}/scripts/run/profile_rs.sh"
    ;;
  train-rs)
    bash "${ROOT}/scripts/run/train_rs.sh"
    ;;
  profile-scp)
    bash "${ROOT}/scripts/run/profile_scp.sh" all
    ;;
  solve-scp)
    bash "${ROOT}/scripts/run/solve_scp.sh"
    ;;
  runtime)
    bash "${ROOT}/scripts/run/run_matrix.sh"
    ;;
  launch-runtime)
    bash "${ROOT}/scripts/run/launch_two_node.sh"
    ;;
  summarize)
    "${PYTHON_BIN:-python3}" "${ROOT}/scripts/analysis/summarize_results.py" \
      --root "${RESULTS_ROOT:?Set RESULTS_ROOT}" \
      --out "${RESULTS_ROOT}/summary"
    ;;
  help|*)
    cat <<'EOF'
Usage: reproduce.sh ACTION

Actions: validate, check-env, reference-figures, profile-rs, train-rs,
profile-scp, solve-scp, runtime, launch-runtime, summarize
EOF
    [[ "${ACTION}" == "help" ]] || exit 2
    ;;
esac
