# Nebius H100 inference box for overversion.com

One H100 SXM VM on Nebius that runs SGLang in deterministic mode behind Caddy at
`https://infer.overversion.com`.

**Costs, so the choice between stopping and deleting is explicit:**

| State | Cost | Time to serving |
|---|---|---|
| Running | $3.85/h plus the disk | seconds |
| Stopped | disk only: 100 GiB network SSD is $11.80/month | ~3 min (`vm.sh start`, `inference.sh up`) |
| Deleted | nothing | ~25 min (`create-vm.sh`, `provision.sh`, `inference.sh up`) |
| Reserved IP | free in every state ([Virtual Networks pricing](https://docs.nebius.com/vpc/resources/pricing)) | saves a DNS change and its propagation wait |

Stopped VMs are not billed for compute. A stopped box keeps the pulled image and
the weights, which is the whole reason to stop rather than delete. Between
demos days apart, delete: everything on that disk comes back from Docker Hub,
Hugging Face and this repo, and the envelope of each run is already saved under
`envelopes/`.

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
   Writes `.instance.env` with the instance id and public IP. The IP comes from
   a reserved allocation named `overversion-infer-ip` (`./ip.sh reserve` creates
   it the first time and reuses it after that), so it survives stop/start and
   delete/create. `DYNAMIC_IP=1 ./create-vm.sh` opts out.
4. **DNS**: A record `infer.overversion.com` to the reserved IP, once. Caddy
   gets the Let's Encrypt certificate on first request, so the record must
   resolve before `./inference.sh up`.
5. **Provision** (about 10 minutes: toolkit check, firewall, image pull, 16 GB
   of Qwen3-8B weights, weights hash, compose and Caddy files):
   ```bash
   ./provision.sh
   ```

## Every demo session

```bash
./vm.sh start          # if stopped; the reserved IP comes back with it
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

## OpenTofu state

Tofu config lives in `tofu/`. Its state is in the `overversion-tfstate` bucket in
Nebius Object Storage. The bucket has versioning on, and S3 lockfile locking
(`use_lockfile`), so two laptops can't apply at the same time. Only the
`tfstate-sa` service account can write to it, through the `tfstate-writers` group
and a bucket policy that gives that group `storage.object-editor`.

```bash
./state-bucket.sh      # once per project: SA, group, bucket, access key into .secrets.env
./tofu.sh init         # then plan / apply / state list ... as usual
```

`.secrets.env` itself is kept in the same bucket at `secrets/nebius.secrets.env`.
`./secrets.sh push` uploads it after you edit it; `./secrets.sh pull` downloads it.
On a new laptop, `nebius profile create` is enough: pull fetches the bucket key
from Nebius IAM with your login, then `./tofu.sh init`. Always use `./tofu.sh`
rather than plain `tofu`: it exports the bucket key for that one process only.

## Files

- `config.env` committed defaults (image tag, model, preset, domain).
- `.secrets.env` API key, HF token, ACME email (ignored).
- `.instance.env` instance id and IP (ignored, written by `create-vm.sh`).
- `create-vm.sh`, `vm.sh` Nebius CLI lifecycle.
- `ip.sh` reserved public IP: `reserve`, `attach` (for a VM created with a dynamic IP), `release`.
- `provision.sh` runs `remote/provision.sh` on the VM over SSH.
- `inference.sh` runs `remote/inference.sh` on the VM over SSH; `health` and `up` also act from here.
- `envelopes/` one JSON per `up`.
- `state-bucket.sh` one-time bootstrap of the OpenTofu state bucket; `tofu.sh` runs `tofu` in `tofu/` with its key.
- `secrets.sh` push/pull `.secrets.env` to the state bucket.

## Gotchas

- Do not run `shutdown` inside the VM; Nebius reboots it and keeps billing. Use `./vm.sh stop`.
- Nebius disks grow but never shrink. Raising `NEBIUS_BOOT_DISK_GIB` is a live
  resize; lowering it means deleting the VM and provisioning again.
- `./vm.sh delete` removes the instance; check `nebius compute disk list`
  afterwards, since a disk left behind keeps billing on its own.
- A dynamic public IP goes back to Nebius after an hour stopped, so a box
  created before `ip.sh` gets a new one on start. `./ip.sh attach` moves it to
  the reserved IP once.
- `./vm.sh delete` leaves the reserved IP in place, on purpose. `./ip.sh release`
  gives it back. The default quota is 3 occupied public IPv4 addresses per
  project, so the reservation uses one of them.
- The image is pinned (`lmsysorg/sglang:v0.5.19-cu129`) and the host driver is 580 (CUDA 13.0), which
  runs CUDA 12.9 containers. Changing either changes the envelope; the saved JSON tells you.
- First `inference.sh up` after provisioning compiles kernels and can take 3 to 5 minutes.
