#!/usr/bin/env bash
# Provision a fresh VM over ssh: toolkit check, firewall, image pull, weights download + hash, compose + Caddy files.
# Safe to re-run. Takes ~10 min on first run (16 GB of weights + 10 GB image).
set -euo pipefail
source "$(dirname "$0")/lib.sh"
[ -n "${SGLANG_API_KEY:-}" ] || die "SGLANG_API_KEY unset; cp .secrets.env.example .secrets.env and fill it"
echo "waiting for ssh on $SSH_USER@$INSTANCE_IP..."
for _ in $(seq 1 60); do ssh_vm -o ConnectTimeout=5 true 2>/dev/null && break; sleep 5; done
ssh_vm_script "$HERE/remote/provision.sh"
