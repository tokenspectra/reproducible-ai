#!/usr/bin/env bash
# Create the Nebius H100 VM with the Nebius CLI. Idempotent-ish: refuses if .instance.env already exists.
# Prereqs (one time): install CLI and log in
#   curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
#   nebius profile create        # browser login; pick the project (parent id) that has H100 quota in eu-north1
set -euo pipefail
source "$(dirname "$0")/lib.sh"
need nebius; need jq
[ -f "$HERE/.instance.env" ] && die ".instance.env exists; delete it (or ./vm.sh delete) before creating another VM"
[ -f "$SSH_KEY_PUB" ] || die "no ssh public key at $SSH_KEY_PUB (set SSH_KEY_PUB in config.env)"

USER_DATA=$(cat <<YAML
#cloud-config
users:
  - name: ${SSH_USER}
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    groups: [docker]
    ssh_authorized_keys:
      - $(cat "$SSH_KEY_PUB")
YAML
)

SUBNET_ID=$(nebius vpc subnet list --format jsonpath='{.items[0].metadata.id}')
[ -n "$SUBNET_ID" ] || die "no subnet in this project; create a VPC network/subnet in the console first"
echo "subnet: $SUBNET_ID"

# Optional static public IP so the infer.<domain> A record survives stop/start.
# Create it once in the console or with `nebius vpc allocation create --help`, then export ALLOCATION_ID before running.
if [ -n "${ALLOCATION_ID:-}" ]; then
  PUBLIC_IP_SPEC="{\"allocation_id\": \"$ALLOCATION_ID\"}"
else
  PUBLIC_IP_SPEC="{}"
  echo "note: dynamic public IP; it can change on stop/start. Update the DNS A record after each start (./vm.sh ip)."
fi

INSTANCE_ID=$(nebius compute instance create \
  --name "$NEBIUS_INSTANCE_NAME" \
  --resources-platform "$NEBIUS_PLATFORM" \
  --resources-preset "$NEBIUS_PRESET" \
  --boot-disk-managed-disk-name "${NEBIUS_INSTANCE_NAME}-disk" \
  --boot-disk-managed-disk-type "$NEBIUS_BOOT_DISK_TYPE" \
  --boot-disk-managed-disk-size-gibibytes "$NEBIUS_BOOT_DISK_GIB" \
  --boot-disk-managed-disk-block-size-bytes 4096 \
  --boot-disk-managed-disk-source-image-family-image-family "$NEBIUS_IMAGE_FAMILY" \
  --boot-disk-attach-mode READ_WRITE \
  --cloud-init-user-data "$USER_DATA" \
  --network-interfaces "[{\"name\": \"eth0\", \"subnet_id\": \"$SUBNET_ID\", \"ip_address\": {}, \"public_ip_address\": $PUBLIC_IP_SPEC}]" \
  --format jsonpath='{.metadata.id}')
echo "instance: $INSTANCE_ID"
printf 'INSTANCE_ID="%s"\n' "$INSTANCE_ID" > "$HERE/.instance.env"

echo "waiting for public IP..."
for _ in $(seq 1 60); do
  IP=$(nebius compute instance get --id "$INSTANCE_ID" --format json \
       | jq -r '.status.network_interfaces[0].public_ip_address.address // empty' | cut -d/ -f1)
  [ -n "$IP" ] && break; sleep 5
done
[ -n "${IP:-}" ] || die "no public IP after 5 minutes; check the console"
printf 'INSTANCE_IP="%s"\n' "$IP" >> "$HERE/.instance.env"
echo "public IP: $IP  (written to .instance.env)"
echo "next: point an A record ${INFER_DOMAIN} -> $IP, wait for ssh, then ./provision.sh"
