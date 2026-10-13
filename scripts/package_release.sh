#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_NAME="$(basename "${ROOT}")"
OUT="${1:-${ROOT}/../SPARSE_code.tar.gz}"

bash "${ROOT}/scripts/validate_release.sh"
rm -f "${OUT}"
tar -C "$(dirname "${ROOT}")" \
  --exclude="${ROOT_NAME}/.env" \
  --exclude="${ROOT_NAME}/.git" \
  --exclude="${ROOT_NAME}/__pycache__" \
  --exclude="${ROOT_NAME}/results/local" \
  -czf "${OUT}" "${ROOT_NAME}"

echo "[package] wrote ${OUT}"
shasum -a 256 "${OUT}"
