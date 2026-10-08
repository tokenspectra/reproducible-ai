# State lives in Nebius Object Storage (S3-compatible), created by ../state-bucket.sh.
# Keep bucket and region in sync with TFSTATE_* in ../config.env.
# Credentials come from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, which ../tofu.sh sets.
terraform {
  required_version = ">= 1.10" # S3 lockfile locking

  backend "s3" {
    bucket = "overversion-tfstate"
    key    = "overversion/terraform.tfstate"
    region = "eu-north1"

    endpoints = { s3 = "https://storage.eu-north1.nebius.cloud" }

    # A .tflock object next to the state; no DynamoDB table needed.
    use_lockfile = true

    # Nebius is not AWS: skip the AWS-only account, region and checksum calls.
    use_path_style              = true
    skip_credentials_validation = true
    skip_region_validation      = true
    skip_requesting_account_id  = true
    skip_metadata_api_check     = true
    skip_s3_checksum            = true
  }
}
