#!/usr/bin/env bash
# Sync .secrets.env with the state bucket, so a new laptop needs only a Nebius login.
#   ./secrets.sh push           upload this laptop's .secrets.env (the bucket keeps old versions)
#   ./secrets.sh pull [--force] download it; refuses to overwrite a different local copy without --force
# The bucket key is read from .secrets.env when present, otherwise from Nebius IAM
# with your own `nebius` login, which is what lets pull work on a fresh machine.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
need curl

OBJECT="secrets/nebius.secrets.env"
URL="https://storage.$TFSTATE_REGION.nebius.cloud/$TFSTATE_BUCKET/$OBJECT"
LOCAL="$HERE/.secrets.env"

if [ -z "${TFSTATE_ACCESS_KEY_ID:-}" ]; then
  need nebius; need jq
  PROJECT_ID=$(nebius config get parent-id)
  SA_ID=$(nebius iam service-account get-by-name --parent-id "$PROJECT_ID" --name tfstate-sa --format jsonpath='{.metadata.id}')
  KEY_ID=$(nebius iam v2 access-key list-by-account --account-service-account-id "$SA_ID" --format json \
    | jq -r '[.items[] | select(.status.state == "ACTIVE")][0].metadata.id // empty')
  [ -n "$KEY_ID" ] || die "tfstate-sa has no active access key; run ./state-bucket.sh"
  KEY_JSON=$(nebius iam v2 access-key get --id "$KEY_ID" --format json)
  TFSTATE_ACCESS_KEY_ID=$(jq -r '.status.aws_access_key_id' <<<"$KEY_JSON")
  TFSTATE_SECRET_ACCESS_KEY=$(jq -r '.status.secret' <<<"$KEY_JSON")
fi

s3() { # s3 <curl args...>
  curl -sSf --aws-sigv4 "aws:amz:$TFSTATE_REGION:s3" \
    --user "$TFSTATE_ACCESS_KEY_ID:$TFSTATE_SECRET_ACCESS_KEY" "$@"
}

case "${1:-}" in
  push)
    [ -f "$LOCAL" ] || die "no $LOCAL to push"
    s3 -X PUT -H "Content-Type: text/plain" --upload-file "$LOCAL" "$URL" >/dev/null
    echo "pushed .secrets.env to s3://$TFSTATE_BUCKET/$OBJECT"
    ;;
  pull)
    tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
    s3 -o "$tmp" "$URL"
    if [ -f "$LOCAL" ] && ! cmp -s "$tmp" "$LOCAL" && [ "${2:-}" != "--force" ]; then
      die "local .secrets.env differs from the bucket copy; diff them, then push or pull --force"
    fi
    install -m 600 "$tmp" "$LOCAL"
    echo "pulled s3://$TFSTATE_BUCKET/$OBJECT into .secrets.env"
    ;;
  *) die "usage: $0 push | pull [--force]" ;;
esac
