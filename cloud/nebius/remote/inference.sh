#!/usr/bin/env bash
# Runs ON the VM. up|down|status|logs|envelope
set -euo pipefail
cd "${DATA_DIR:?}/overversion"
cmd="${1:-status}"
PROFILES=(); [ "${RUN_BASELINE:-0}" = "1" ] && PROFILES=(--profile baseline)
case "$cmd" in
  up)
    sudo docker compose "${PROFILES[@]}" up -d
    echo "waiting for sglang-det health (model load takes 1-3 min)..."
    for _ in $(seq 1 120); do
      s=$(sudo docker inspect --format '{{.State.Health.Status}}' sglang-det 2>/dev/null || echo starting)
      [ "$s" = healthy ] && break; sleep 5
    done
    [ "$s" = healthy ] || { echo "not healthy: $s"; sudo docker logs --tail 50 sglang-det; exit 1; }
    echo "sglang-det healthy"; ./envelope.sh ;;
  down)   sudo docker compose --profile baseline down ;;
  status) sudo docker compose --profile baseline ps ;;
  logs)   sudo docker logs --tail "${2:-100}" -f sglang-det ;;
  envelope) ./envelope.sh ;;
  *) echo "usage: up|down|status|logs [n]|envelope"; exit 1 ;;
esac
