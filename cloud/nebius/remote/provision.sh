#!/usr/bin/env bash
# Runs ON the VM (piped over ssh by ../provision.sh). Idempotent. Env comes from lib.sh.
set -euo pipefail
: "${SGLANG_IMAGE:?}" "${MODEL_ID:?}" "${INFER_DOMAIN:?}" "${DATA_DIR:?}" "${SGLANG_API_KEY:?}"
echo "== host facts"
nvidia-smi --query-gpu=name,driver_version,pci.device_id --format=csv,noheader || { echo "nvidia-smi failed: wrong image?"; exit 1; }
command -v docker >/dev/null || { echo "docker missing on this image"; exit 1; }
sudo docker compose version >/dev/null 2>&1 || { sudo apt-get update -qq && sudo apt-get install -y -qq docker-compose-plugin; }

echo "== docker group + nvidia toolkit"
sudo usermod -aG docker "$USER" || true
if ! docker info 2>/dev/null | grep -q -i 'nvidia'; then
  if ! command -v nvidia-ctk >/dev/null; then
    echo "installing nvidia-container-toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
  fi
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
fi

echo "== firewall: 22, 80, 443 only (sglang ports are never published on the host)"
sudo apt-get install -y -qq ufw >/dev/null
sudo ufw allow 22/tcp >/dev/null; sudo ufw allow 80/tcp >/dev/null; sudo ufw allow 443/tcp >/dev/null
sudo ufw --force enable >/dev/null

echo "== layout under $DATA_DIR"
sudo mkdir -p "$DATA_DIR"/{hf,overversion,caddy}; sudo chown -R "$USER:$USER" "$DATA_DIR"

echo "== pull $SGLANG_IMAGE"
sudo docker pull -q "$SGLANG_IMAGE"

echo "== download $MODEL_ID into $DATA_DIR/hf (skips if present)"
sudo docker run --rm -e HF_TOKEN="${HF_TOKEN:-}" -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -v "$DATA_DIR/hf:/root/.cache/huggingface" "$SGLANG_IMAGE" \
  bash -lc "pip install -q -U huggingface_hub[cli] hf_transfer >/dev/null 2>&1 || true; hf download '$MODEL_ID' --exclude '*.gguf' || huggingface-cli download '$MODEL_ID' --exclude '*.gguf'" \
  | tail -1

echo "== weights hash (once; part of the envelope)"
# The HF cache is written by a root container, so the snapshot directory is not
# writable by us. The manifest therefore lives beside the compose files, which
# we own, and records the snapshot id it was taken from.
SNAP=$(sudo ls -d "$DATA_DIR"/hf/hub/models--${MODEL_ID//\//--}/snapshots/*/ | head -1)
MANIFEST="$DATA_DIR/overversion/weights.sha256"
if [ ! -s "$MANIFEST" ]; then
  echo "# model=$MODEL_ID snapshot=$(basename "$SNAP")" > "$MANIFEST"
  sudo sh -c "cd '$SNAP' && sha256sum -- *.safetensors *.json 2>/dev/null | sort -k2" >> "$MANIFEST"
fi
echo "weights manifest: $MANIFEST ($(($(wc -l < "$MANIFEST") - 1)) files)"

echo "== compose + caddy config"
MEM_FRAC="0.85"; [ "${RUN_BASELINE:-0}" = "1" ] && MEM_FRAC="0.42"
cat > "$DATA_DIR/overversion/.env" <<ENV
SGLANG_IMAGE=$SGLANG_IMAGE
MODEL_ID=$MODEL_ID
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-model}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-fa3}
SGLANG_API_KEY=$SGLANG_API_KEY
INFER_DOMAIN=$INFER_DOMAIN
ACME_EMAIL=${ACME_EMAIL:-}
MEM_FRAC=$MEM_FRAC
DATA_DIR=$DATA_DIR
ENV
chmod 600 "$DATA_DIR/overversion/.env"

cat > "$DATA_DIR/overversion/docker-compose.yml" <<'YML'
x-sglang: &sglang
  image: ${SGLANG_IMAGE}
  restart: unless-stopped
  ipc: host
  shm_size: "32g"
  deploy: { resources: { reservations: { devices: [ { driver: nvidia, count: all, capabilities: [gpu] } ] } } }
  volumes: [ "${DATA_DIR}/hf:/root/.cache/huggingface" ]
  environment: [ "HF_HUB_OFFLINE=1" ]
  healthcheck:
    test: ["CMD-SHELL", "curl -fsS -H 'Authorization: Bearer ${SGLANG_API_KEY}' http://localhost:30000/health || exit 1"]
    interval: 15s
    timeout: 5s
    retries: 80
    start_period: 120s

services:
  sglang:
    <<: *sglang
    container_name: sglang-det
    command: >
      python3 -m sglang.launch_server
        --model-path ${MODEL_ID} --served-model-name ${SERVED_MODEL_NAME}
        --host 0.0.0.0 --port 30000 --api-key ${SGLANG_API_KEY}
        --enable-deterministic-inference --attention-backend ${ATTENTION_BACKEND}
        --mem-fraction-static ${MEM_FRAC} --tp 1
  sglang-baseline:
    <<: *sglang
    container_name: sglang-baseline
    profiles: ["baseline"]
    command: >
      python3 -m sglang.launch_server
        --model-path ${MODEL_ID} --served-model-name ${SERVED_MODEL_NAME}
        --host 0.0.0.0 --port 30000 --api-key ${SGLANG_API_KEY}
        --attention-backend ${ATTENTION_BACKEND}
        --mem-fraction-static ${MEM_FRAC} --tp 1
  caddy:
    image: caddy:2
    container_name: caddy
    restart: unless-stopped
    ports: ["80:80", "443:443"]
    environment: [ "INFER_DOMAIN=${INFER_DOMAIN}", "ACME_EMAIL=${ACME_EMAIL}" ]
    volumes:
      - "${DATA_DIR}/overversion/Caddyfile:/etc/caddy/Caddyfile:ro"
      - "${DATA_DIR}/caddy:/data"
YML

cat > "$DATA_DIR/overversion/Caddyfile" <<'CADDY'
{
	email {$ACME_EMAIL}
}
{$INFER_DOMAIN} {
	encode zstd gzip
	# baseline (non-deterministic) instance, only when started with the "baseline" profile
	handle_path /baseline/* {
		reverse_proxy sglang-baseline:30000 {
			flush_interval -1
		}
	}
	handle {
		reverse_proxy sglang:30000 {
			flush_interval -1
		}
	}
}
CADDY

echo "== envelope helper"
cat > "$DATA_DIR/overversion/envelope.sh" <<'ENVL'
#!/usr/bin/env bash
# Print the hardware/software envelope of this box as JSON. Run after `docker compose up` for the sglang fields.
set -euo pipefail
cd "$(dirname "$0")"; set -a; source .env; set +a
q() { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits | head -1 | xargs; }
SNAP=$(sudo ls -d "$DATA_DIR"/hf/hub/models--${MODEL_ID//\//--}/snapshots/*/ | head -1)
MANIFEST="$DATA_DIR/overversion/weights.sha256"
DIGEST=$(sudo docker inspect --format '{{index .RepoDigests 0}}' "$SGLANG_IMAGE" 2>/dev/null || echo "")
INFO=$(curl -fsS -H "Authorization: Bearer $SGLANG_API_KEY" http://localhost:30000/server_info 2>/dev/null \
       || sudo docker exec sglang-det curl -fsS -H "Authorization: Bearer $SGLANG_API_KEY" http://localhost:30000/server_info 2>/dev/null || echo '{}')
jq -n \
  --arg gpu_name "$(q name)" --arg driver "$(q driver_version)" --arg pci "$(q pci.device_id)" \
  --arg vbios "$(q vbios_version)" --arg cuda "$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | cut -d' ' -f3)" \
  --arg kernel "$(uname -r)" --arg os "$(. /etc/os-release && echo "$PRETTY_NAME")" \
  --arg image "$SGLANG_IMAGE" --arg digest "$DIGEST" --arg model "$MODEL_ID" \
  --arg snapshot "$(basename "$SNAP")" --arg weights_manifest_sha "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" \
  --arg attn "$ATTENTION_BACKEND" --arg hostname "$(hostname)" --arg captured "$(date -u +%FT%TZ)" \
  --argjson server_info "$INFO" \
  '{captured_at:$captured, hostname:$hostname, gpu:{name:$gpu_name, arch:"sm_90", pci_device_id:$pci, driver:$driver, cuda_driver:$cuda, vbios:$vbios},
    os:$os, kernel:$kernel, sglang:{image:$image, digest:$digest, version:($server_info.version // null), deterministic:($server_info.enable_deterministic_inference // null), attention_backend:$attn},
    model:{id:$model, snapshot:$snapshot, weights_manifest_sha256:$weights_manifest_sha}}'
ENVL
chmod +x "$DATA_DIR/overversion/envelope.sh"
echo "== provision done. next: ./inference.sh up"
