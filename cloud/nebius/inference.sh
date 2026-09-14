#!/usr/bin/env bash
# Control the inference stack on the VM from here: up | down | status | logs | envelope | health
set -euo pipefail
source "$(dirname "$0")/lib.sh"
cmd="${1:-status}"
case "$cmd" in
  up)
    ssh_vm_script "$HERE/remote/inference.sh" up | tee /tmp/overversion-envelope.txt
    f="$HERE/envelopes/$(date -u +%Y%m%dT%H%M%SZ).json"
    sed -n '/^{/,$p' /tmp/overversion-envelope.txt > "$f" && echo "envelope saved: $f"
    echo "public endpoint: https://$INFER_DOMAIN  (Caddy fetches the certificate on first request; needs the A record to resolve to $INSTANCE_IP)" ;;
  health)
    curl -fsS -m 10 "https://$INFER_DOMAIN/health" && echo " det: up" || echo "det: DOWN"
    [ "$RUN_BASELINE" = "1" ] && { curl -fsS -m 10 "https://$INFER_DOMAIN/baseline/health" && echo " baseline: up" || echo "baseline: DOWN"; } ;;
  envelope) ssh_vm_script "$HERE/remote/inference.sh" envelope ;;
  down|status|logs) ssh_vm_script "$HERE/remote/inference.sh" "$@" ;;
  *) die "usage: $0 up|down|status|logs [n]|envelope|health" ;;
esac
