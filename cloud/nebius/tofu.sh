#!/usr/bin/env bash
# Run OpenTofu in tofu/ with the state bucket key from .secrets.env, e.g. ./tofu.sh init, ./tofu.sh plan.
# The key is exported only for this process so it never shadows other AWS credentials.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
need tofu
[ -n "${TFSTATE_ACCESS_KEY_ID:-}" ] || die "TFSTATE_ACCESS_KEY_ID unset; run ./state-bucket.sh or copy .secrets.env from the other laptop"
export AWS_ACCESS_KEY_ID="$TFSTATE_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$TFSTATE_SECRET_ACCESS_KEY"
exec tofu -chdir="$HERE/tofu" "$@"
