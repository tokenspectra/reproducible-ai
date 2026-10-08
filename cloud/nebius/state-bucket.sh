#!/usr/bin/env bash
# One-time bootstrap of the remote OpenTofu state: a versioned bucket that only a
# dedicated service account can write, plus an S3 access key for it appended to
# .secrets.env. Safe to re-run: existing pieces are reused, and a key is minted
# only when .secrets.env has none.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
need nebius; need jq

PROJECT_ID=$(nebius config get parent-id)
TENANT_ID=$(nebius iam project get "$PROJECT_ID" --format jsonpath='{.metadata.parent_id}')
SA_NAME="tfstate-sa"
GROUP_NAME="tfstate-writers"

get_id() { "$@" --format jsonpath='{.metadata.id}' 2>/dev/null || true; }

SA_ID=$(get_id nebius iam service-account get-by-name --parent-id "$PROJECT_ID" --name "$SA_NAME")
[ -n "$SA_ID" ] || SA_ID=$(get_id nebius iam service-account create --parent-id "$PROJECT_ID" --name "$SA_NAME")
echo "service account: $SA_ID"

# Bucket policies grant roles to groups, so the SA gets a group of its own rather
# than joining the tenant-wide editors group.
GROUP_ID=$(get_id nebius iam group get-by-name --parent-id "$TENANT_ID" --name "$GROUP_NAME")
[ -n "$GROUP_ID" ] || GROUP_ID=$(get_id nebius iam group create --parent-id "$TENANT_ID" --name "$GROUP_NAME")
echo "group: $GROUP_ID"
if ! nebius iam group-membership list-members --parent-id "$GROUP_ID" --format json \
    | jq -e --arg sa "$SA_ID" '[.memberships[]?.spec.member_id] | index($sa)' >/dev/null; then
  nebius iam group-membership create --parent-id "$GROUP_ID" --member-id "$SA_ID" >/dev/null
fi

BUCKET_ID=$(get_id nebius storage bucket get-by-name --parent-id "$PROJECT_ID" --name "$TFSTATE_BUCKET")
if [ -z "$BUCKET_ID" ]; then
  # Versioning keeps every past state, so a bad apply or a corrupted write is recoverable.
  BUCKET_ID=$(get_id nebius storage bucket create --parent-id "$PROJECT_ID" --name "$TFSTATE_BUCKET" \
    --versioning-policy enabled \
    --bucket-policy-rules "[{\"group_id\":\"$GROUP_ID\",\"paths\":[\"*\"],\"roles\":[\"storage.object-editor\"]}]")
fi
echo "bucket: $TFSTATE_BUCKET ($BUCKET_ID)"

if [ -n "${TFSTATE_ACCESS_KEY_ID:-}" ]; then
  echo "TFSTATE_ACCESS_KEY_ID already in .secrets.env; not minting another key"
  exit 0
fi
KEY_ID=$(get_id nebius iam v2 access-key create --parent-id "$PROJECT_ID" \
  --account-service-account-id "$SA_ID" --description "OpenTofu remote state")
KEY_JSON=$(nebius iam v2 access-key get --id "$KEY_ID" --format json)
{
  echo ""
  echo "# OpenTofu state bucket key for $SA_NAME (written by state-bucket.sh)"
  echo "TFSTATE_ACCESS_KEY_ID=$(jq -r '.status.aws_access_key_id' <<<"$KEY_JSON")"
  echo "TFSTATE_SECRET_ACCESS_KEY=$(jq -r '.status.secret' <<<"$KEY_JSON")"
} >> "$HERE/.secrets.env"
echo "access key $KEY_ID appended to .secrets.env"
