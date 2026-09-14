# shellcheck shell=bash
# Shared helpers. Sourced by the other scripts; not executable on its own.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.env"
[ -f "$HERE/.secrets.env" ] && source "$HERE/.secrets.env"
[ -f "$HERE/.instance.env" ] && source "$HERE/.instance.env"

die() { echo "error: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

require_instance() {
  [ -n "${INSTANCE_IP:-}" ] || die "no INSTANCE_IP; run ./vm.sh ip (or write cloud/nebius/.instance.env with INSTANCE_IP=...)"
}

ssh_vm() { # ssh_vm <cmd...>
  require_instance
  ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$SSH_USER@$INSTANCE_IP" "$@"
}

ssh_vm_script() { # ssh_vm_script <local script path> [args...] ; runs it remotely with env from config
  require_instance
  local script="$1"; shift
  ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$SSH_USER@$INSTANCE_IP" \
    "SGLANG_IMAGE='$SGLANG_IMAGE' MODEL_ID='$MODEL_ID' SERVED_MODEL_NAME='$SERVED_MODEL_NAME' \
     ATTENTION_BACKEND='$ATTENTION_BACKEND' INFER_DOMAIN='$INFER_DOMAIN' RUN_BASELINE='$RUN_BASELINE' \
     DATA_DIR='$DATA_DIR' SGLANG_API_KEY='${SGLANG_API_KEY:-}' HF_TOKEN='${HF_TOKEN:-}' ACME_EMAIL='${ACME_EMAIL:-}' \
     bash -s -- $*" < "$script"
}
