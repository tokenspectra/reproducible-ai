#!/usr/bin/env bash
# Reserved public IP for the VM, so the infer.<domain> A record survives stop/start and delete/create.
#   ./ip.sh reserve   find or create the allocation; prints "<allocation id> <ip>"
#   ./ip.sh attach    put it on the VM in .instance.env (one-off, for a VM created with a dynamic IP)
#   ./ip.sh release   delete the allocation; the address goes back to Nebius
# An allocation outlives the VM: ./vm.sh delete leaves it in place, which is the point.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
need nebius; need jq
cmd="${1:-reserve}"

find_allocation() { # prints the allocation JSON, or nothing
  nebius vpc allocation list --format json \
    | jq -c --arg n "$NEBIUS_IP_ALLOCATION_NAME" '.items[]? | select(.metadata.name == $n)'
}

reserve() { # prints "<allocation id> <ip>"
  local a
  a=$(find_allocation)
  if [ -z "$a" ]; then
    local subnet
    subnet=$(nebius vpc subnet list --format jsonpath='{.items[0].metadata.id}')
    [ -n "$subnet" ] || die "no subnet in this project; create a VPC network/subnet in the console first"
    nebius vpc allocation create --name "$NEBIUS_IP_ALLOCATION_NAME" --ipv4-public-subnet-id "$subnet" >/dev/null
    a=$(find_allocation)
  fi
  [ -n "$a" ] || die "allocation $NEBIUS_IP_ALLOCATION_NAME not found after create"
  jq -r '"\(.metadata.id) \(.status.details.allocated_cidr | split("/")[0])"' <<<"$a"
}

case "$cmd" in
  reserve) reserve ;;
  attach)
    [ -n "${INSTANCE_ID:-}" ] || die "no INSTANCE_ID in .instance.env"
    read -r id ip < <(reserve)
    nebius compute instance update --id "$INSTANCE_ID" \
      --network-interfaces "[{\"public_ip_address\": {\"allocation_id\": \"$id\"}}]"
    sed -i '/^INSTANCE_IP=/d' "$HERE/.instance.env"; printf 'INSTANCE_IP="%s"\n' "$ip" >> "$HERE/.instance.env"
    echo "attached $ip ($id); point ${INFER_DOMAIN} at it once and leave it" ;;
  release)
    a=$(find_allocation); [ -n "$a" ] || die "no allocation named $NEBIUS_IP_ALLOCATION_NAME"
    read -r -p "release $(jq -r '.status.details.allocated_cidr' <<<"$a")? DNS for ${INFER_DOMAIN} will point nowhere. type yes: " ans
    [ "$ans" = yes ] || exit 1
    nebius vpc allocation delete --id "$(jq -r .metadata.id <<<"$a")"; echo "released" ;;
  *) die "usage: $0 reserve|attach|release" ;;
esac
