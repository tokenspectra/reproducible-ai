#!/usr/bin/env bash
# Lifecycle of the Nebius VM: start | stop | status | ip | ssh | delete
# Stopped VMs are not billed for compute (the boot disk still is). Never run `shutdown` inside the guest:
# Nebius treats that as a failure and reboots it, and billing continues.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
need nebius; need jq
cmd="${1:-status}"
[ -n "${INSTANCE_ID:-}" ] || die "no INSTANCE_ID in .instance.env; run ./create-vm.sh or write it by hand"

refresh_ip() {
  IP=$(nebius compute instance get --id "$INSTANCE_ID" --format json \
       | jq -r '.status.network_interfaces[0].public_ip_address.address // empty' | cut -d/ -f1)
  if [ -n "$IP" ]; then
    sed -i '/^INSTANCE_IP=/d' "$HERE/.instance.env"; printf 'INSTANCE_IP="%s"\n' "$IP" >> "$HERE/.instance.env"
  fi
  echo "${IP:-<no public ip>}"
}

case "$cmd" in
  status) nebius compute instance get --id "$INSTANCE_ID" --format json | jq '{name: .metadata.name, state: .status.state, ip: .status.network_interfaces[0].public_ip_address.address}' ;;
  ip)     refresh_ip ;;
  start)  nebius compute instance start --id "$INSTANCE_ID"; echo "started; ip: $(refresh_ip)"; echo "if the IP changed, update the ${INFER_DOMAIN} A record before ./inference.sh up" ;;
  stop)   nebius compute instance stop --id "$INSTANCE_ID"; echo "stopped (compute billing off; disk billing continues)" ;;
  ssh)    shift; ssh_vm "$@" ;;
  delete) read -r -p "delete instance $INSTANCE_ID and its boot disk (weights, images)? type yes: " a; [ "$a" = yes ] || exit 1
          nebius compute instance delete --id "$INSTANCE_ID"; rm -f "$HERE/.instance.env"; echo "deleted (disk may need separate deletion: nebius compute disk list)" ;;
  *) die "usage: $0 start|stop|status|ip|ssh [cmd]|delete" ;;
esac
