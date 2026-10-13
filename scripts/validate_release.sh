#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYCACHE_ROOT="$(mktemp -d)"
export PYTHONPYCACHEPREFIX="${PYCACHE_ROOT}"
trap 'rm -rf "${PYCACHE_ROOT}"' EXIT

required=(README.md LICENSE .env.example requirements.txt scripts/perf.py scripts/run/run_case.sh scripts/run/run_matrix.sh)
for path in "${required[@]}"; do
  [[ -s "${path}" ]] || { echo "[validate][ERROR] missing ${path}" >&2; exit 2; }
done

while IFS= read -r script; do
  bash -n "${script}"
done < <(find scripts -name '*.sh' -type f | sort)

"${PYTHON_BIN}" -m compileall -q scripts utils tests
"${PYTHON_BIN}" -m unittest discover -s tests -p 'test_*.py'

if rg -n '/Users/|/home/|10\.[0-9]+\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|BEGIN [A-Z ]*PRIVATE KEY|ssh-(rsa|ed25519)|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}' \
  --glob '!scripts/validate_release.sh' \
  --glob '!docs/RELEASE_CHECKLIST.md' \
  .; then
  echo "[validate][ERROR] possible identifying or private infrastructure string found." >&2
  exit 2
fi

if find . -type f \( -name '*.pth' -o -name '*.pt' -o -name '*.safetensors' -o -name '*.bin' -o -name '*.log' -o -name '*.jsonl.gz' \) -print | grep -q .; then
  echo "[validate][ERROR] model, dataset, checkpoint, or raw-log file found." >&2
  exit 2
fi

large_files="$(find . -type f -size +5M -print)"
if [[ -n "${large_files}" ]]; then
  echo "[validate][ERROR] files larger than 5 MiB are not allowed:" >&2
  echo "${large_files}" >&2
  exit 2
fi

echo "[validate] PASS"
