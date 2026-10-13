#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
# shellcheck disable=SC1090
source "${ENV_FILE}"

ACTION="${1:-show}"
IFACE="${NETWORK_INTERFACE:-${NCCL_SOCKET_IFNAME:-}}"
RATE="${RATE_MBIT:-400}"
[[ -n "${IFACE}" ]] || { echo "[network][ERROR] NETWORK_INTERFACE is required." >&2; exit 2; }

case "${ACTION}" in
  show)
    tc qdisc show dev "${IFACE}"
    ;;
  native)
    sudo tc qdisc del dev "${IFACE}" root 2>/dev/null || true
    tc qdisc show dev "${IFACE}"
    ;;
  limit)
    sudo tc qdisc replace dev "${IFACE}" root tbf \
      rate "${RATE}mbit" burst "${TC_BURST:-2mb}" latency "${TC_LATENCY:-100ms}"
    tc qdisc show dev "${IFACE}"
    ;;
  *)
    echo "Usage: configure_network.sh show|native|limit" >&2
    exit 2
    ;;
esac
