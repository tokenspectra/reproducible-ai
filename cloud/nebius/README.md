# Nebius H100 inference box for overversion.com

One H100 SXM VM on Nebius that runs SGLang in deterministic mode behind Caddy at
`https://infer.overversion.com`. It is **up only when you start it**: stopped VMs
are not billed for compute (about $3.85/h running, about $14/month for the
200 GiB boot disk while stopped). The boot disk keeps the pulled image and the
model weights, so start-to-serving is a few minutes.

Everything runs from this laptop over SSH. Nothing here needs the Nebius
console after the first login.

## One-time

1. **Nebius account** with H100 quota (default quota is 32 H100 in `eu-north1`,
   which is enough). Install and log in to the CLI:
   ```bash
   curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
   nebius profile create        # browser login, choose the project as parent
   nebius vpc subnet list       # must show one subnet; create a network in the console if empty
   ```
2. **Secrets**: `cp .secrets.env.example .secrets.env`, set `SGLANG_API_KEY`
   (`openssl rand -hex 32`) and `ACME_EMAIL`. Never commit it.
3. **Create the VM** (about 2 minutes):
   ```bash
   ./create-vm.sh
   ```
   Writes `.instance.env` with the instance id and public IP. For a stable IP
   across stop/start, create a public IP allocation once (console, or
   `nebius vpc allocation create --help`) and run `ALLOCATION_ID=... ./create-vm.sh`.
4. **DNS**: A record `infer.overversion.com` to the printed IP. Caddy gets the
   Let's Encrypt certificate on first request, so the record must resolve
   before `./inference.sh up`.
5. **Provision** (about 10 minutes: toolkit check, firewall, image pull, 16 GB
   of Qwen3-8B weights, weights hash, compose and Caddy files):
   ```bash
   ./provision.sh
   ```

## Every demo session

```bash
./vm.sh start          # if stopped; prints the IP, update DNS if it changed
./inference.sh up      # starts sglang-det (+ baseline), waits for health, saves the envelope JSON
./inference.sh health  # from outside, through Caddy
# ... demo ...
./inference.sh down    # optional; stopping the VM also stops it
./vm.sh stop           # compute billing off
```

`./inference.sh up` saves the envelope (GPU, driver, CUDA, image digest, SGLang
version, weights manifest hash, attention backend) under `envelopes/`. The
Railway service reads the same fields from `/server_info` and shows the
envelope id in the UI.

## Endpoints

| URL | What |
|---|---|
| `https://infer.overversion.com/v1/chat/completions` | deterministic instance, OpenAI-compatible, `Authorization: Bearer $SGLANG_API_KEY` |
| `https://infer.overversion.com/server_info` | envelope fields, including `enable_deterministic_inference` |
| `https://infer.overversion.com/baseline/v1/...` | non-deterministic instance for the comparison panel (only with `RUN_BASELINE=1`) |

SGLang ports are never published on the host; only Caddy's 80 and 443 and SSH
are open (ufw). Both SGLang instances share the GPU at 42 percent memory each
when the baseline is on, 85 percent otherwise.

## Deterministic-mode facts that matter for the client

- Send an explicit `seed` per request. SGLang defaults it to 42 and does not
  bump it for `n > 1`.
- `min_p` together with a seed is rejected by SGLang. Use temperature, top_p, top_k.
- Full-vocabulary evidence: request `top_logprobs` with a large `top_logprobs_num`
  (no cap) or `token_ids_logprob`. Returned logprobs are temperature-scaled
  unless the server sets `SGLANG_RETURN_ORIGINAL_LOGPROB=1`.
- Radix cache is on with the FA3 backend. Prefill-with-cache equality is an
  open upstream item; the verify path in our service re-prefills the whole
  transcript anyway.

## Files

- `config.env` committed defaults (image tag, model, preset, domain).
- `.secrets.env` API key, HF token, ACME email (ignored).
- `.instance.env` instance id and IP (ignored, written by `create-vm.sh`).
- `create-vm.sh`, `vm.sh` Nebius CLI lifecycle.
- `provision.sh` runs `remote/provision.sh` on the VM over SSH.
- `inference.sh` runs `remote/inference.sh` on the VM over SSH; `health` and `up` also act from here.
- `envelopes/` one JSON per `up`.

## Gotchas

- Do not run `shutdown` inside the VM; Nebius reboots it and keeps billing. Use `./vm.sh stop`.
- Dynamic public IPs can change on start. `./vm.sh start` prints the new one.
- The image is pinned (`lmsysorg/sglang:v0.5.19-cu129`) and the host driver is 580 (CUDA 13.0), which
  runs CUDA 12.9 containers. Changing either changes the envelope; the saved JSON tells you.
- First `inference.sh up` after provisioning compiles kernels and can take 3 to 5 minutes.
