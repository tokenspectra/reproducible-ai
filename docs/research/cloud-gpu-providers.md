# Cloud GPU providers for short single-GPU validation runs (H100 + Blackwell)

**Research date:** 2026-09-09. All prices USD, on-demand list price, observed on the provider's own pricing page / pricing API / docs unless marked *(3rd-party)*. GPU cloud prices move monthly (several providers raised H100 rates during 2025-26); re-check before a run.

**Use case:** solo researcher, deterministic-LLM-inference project. Needs, for a few hours at a time:

- 1x H100 80 GB (SXM or PCIe) and, separately, 1x B200 (or other Blackwell) to test **nvfp4**, which needs Blackwell.
- Root SSH (or at least root in a container), free choice of CUDA/PyTorch image or ability to install vLLM/SGLang, a persistent volume for model weights, per-minute/second billing, no commitment, credit-card signup, no sales call.
- Nice to have: 2x/4x H100 in one node for a later tensor-parallel phase; a reproducible way to record exact GPU SKU, driver and CUDA version.

---

## 1. TL;DR

| Role | Pick | Why |
|---|---|---|
| **Primary, H100** | **Lambda Cloud** — 1x H100 PCIe $3.29/hr, 1x H100 SXM $4.29/hr | Real Ubuntu VM with sudo, Docker + NVIDIA Container Toolkit preinstalled, per-minute billing, card signup, 2x/4x H100 SXM nodes in the same catalog for the TP phase, no egress fees. |
| **Primary, Blackwell** | **Lambda Cloud** — 1x B200 SXM6 $6.99/hr (same account, same image, same tooling). Drop-in alternative: **Verda (ex-DataCrunch)** 1x B200 $6.11/hr, spot $3.06/hr. | Keeping Hopper and Blackwell runs on one provider keeps one software envelope (same Lambda Stack image, same driver family) — the diff between runs is then the GPU, not the cloud. Verda is the cheapest *true root VM* single B200 and is where to go when Lambda has no 1x B200 in stock. |
| **Fallback** | **RunPod Secure Cloud** — H100 PCIe $2.89 / SXM $3.49, B200 $6.79, RTX PRO 6000 Blackwell $2.09 | Per-second billing, broadest Blackwell menu (B200, B300, RTX PRO 6000, RTX 5090), network volumes $0.07/GB-mo. It is a container (root inside, no docker-in-docker) and the driver is whatever the host has, so it is a fallback, not the reference environment. |

**Not recommended for this use case:** CoreWeave (sales-gated, 8x only, Kubernetes only), Crusoe (self-serve now but H100 is 8x only, Blackwell sales-only), Together (8-GPU minimum, 1 TiB storage floor), the big three (single Blackwell not sold at all; 1x H100 $6.9/hr; GPU quota starts at 0 and needs a request), Fireworks/Baseten (no shell). Vast.ai is the cheapest H100 but is a poor fit for a reproducibility envelope (marketplace hosts, volumes tied to one physical machine, ~3 single-B200 offers on the whole market).

**Campaign estimate (20 h H100 + 10 h B200 + 50 GB for 2 months):** about **$155-$180 on Lambda** ($156 with H100 PCIe, $176 with H100 SXM), **~$146 on Verda** ($83 if all spot), **~$145 on RunPod Secure**. Budget **~$250** to cover setup/debug time and a second filesystem if the H100 and B200 land in different Lambda regions. Details in section 5.

---

## 2. Comparison table

Prices are per GPU-hour for a single-GPU instance unless stated. "Node 2x/4x H100" = whether a single node with 2 or 4 H100s is orderable on demand, with the node price.

| Provider | 1x H100 80 GB | 1x Blackwell (SKU) | Node 2x / 4x H100 | Billing | Persistent storage | Root / SSH | Card self-serve, no sales | Main gotchas |
|---|---|---|---|---|---|---|---|---|
| **Lambda Cloud** | PCIe **$3.29**, SXM **$4.29** | **B200 SXM6 $6.99** (2x $6.89, 4x $6.79, 8x $6.69 /GPU). No RTX PRO / 5090 / GB200. | SXM: 2x $8.38, 4x $16.36 (PCIe 1x only) | Per-minute; postpaid weekly invoice | "Filesystems" (NFS) $0.20/GiB-mo, survives termination, **locked to one region**, must attach at launch | Full VM, `ubuntu` + sudo; Docker + NVIDIA Container Toolkit preinstalled | Yes (credit card only, $10 pre-auth; no debit/prepaid); country restrictions | 1x H100 frequently sold out *(3rd-party)*; **no stop/pause** — terminate or pay; new-account instance cap that grows with paid invoices; no spot tier; no egress fees |
| **Verda (ex-DataCrunch)** | SXM5 **$3.25** (spot $1.63); no PCIe | **B200 SXM6 180 GB $6.11** (spot $3.06); B300 $7.50; GB300 $8.62; H200 $4.00 | Linear: 2x $6.50, 4x $13.00, 8x $26.00 (NVLink) | **10-minute prepaid increments** (unused part refunded) | Block volume or shared FS $0.20/GiB-mo; kept by default when instance deleted; 96 h trash | Full VM, **`ssh root@ip`**; images `ubuntu-24.04-cuda-12.8-open-docker`, `-cuda-13.0-open-docker` (Docker preinstalled) | Yes (card/debit, prepaid balance top-up) | **Shut-down instances keep billing** (hibernate/delete instead); balance at $0 deletes volumes after 96 h; spot evicted without warning; Finland/Iceland only; prices rose from ~$2.29 (2025) |
| **RunPod Secure Cloud** | PCIe **$2.89**, SXM **$3.49**, NVL 94 GB $3.19 | **B200 $6.79**; B300 $7.89; **RTX PRO 6000 Blackwell SE $2.09**; RTX 5090 $0.99 | 1-8 GPUs per pod, linear (2x SXM ~$6.98, 4x ~$13.96), subject to host | **Per-second** (network volumes hourly); prepaid, min $10 credit | Network Volume $0.07/GB-mo (<1 TB), $0.14 high-perf; per-datacenter; Secure Cloud only. Pod volume $0.10 running / $0.20 stopped, **erased on terminate** | Container, root inside, `nvidia-smi` works; **no docker-in-docker**; any registry image or `runpod/pytorch` templates; public IP on Secure | Yes (Stripe or crypto); default spend cap $80/hr | Stopped pod may not restart ("Zero GPU") if host GPU rented out; driver = host's (filter by CUDA version); no egress fees |
| **RunPod Community Cloud** | PCIe **$1.99**, SXM **$2.69** | B200 $5.98; RTX PRO 6000 $1.69; RTX 5090 $0.69 | Same mechanism, thinner supply | Per-second | **No network volumes** (pod volume only, lost on terminate) | Same as Secure; public IP not guaranteed (proxied SSH, no SCP) | Yes | Third-party hosts, pool shrinking (no new hosts accepted); availability varies |
| **Vast.ai** | Verified hosts SXM $1.87-5.72 (median $2.41), PCIe $2.54-2.67; interruptible bids from ~$0.67 (live 2026-09-09) | **B200: 3 offers, $5.31 / $7.51 / $10.00**; RTX PRO 6000 from $1.33; RTX 5090 from $0.40 | 2x SXM from $4.54, 4x $9.07 (very thin) | Per-second, min deposit $5 | Instance disk host-set ($0.13-0.20/GB-mo, some B200 hosts $0.94); "Volumes" **tied to the physical host**; storage billed while stopped | Container, root SSH, `nvidia-smi` OK; no docker-in-docker; KVM VM instances on some hosts | Yes (Stripe/crypto), new-account spend-rate limit | Untrusted hosts unless "Secure Cloud" filter; per-host bandwidth fees ($0-0.05/GB); prices/stock change minute to minute; interruptible = killed when outbid |
| **Nebius AI Cloud** | SXM **$3.85** (preemptible $2.15); no PCIe | **B200 $7.15** (preemptible $3.95); B300 $7.85; **RTX PRO 6000 $1.80**; GB200 sales only | **No** — presets are 1 GPU or 8 GPUs only (8x $30.80) | Per-second, prepaid, min top-up $25 | Network SSD $0.071/GiB-mo (non-replicated $0.053), shared FS $0.08; standalone disks survive VM deletion; stopped VMs not billed for compute | Full VM, cloud-init user with NOPASSWD sudo (root login disabled); images document driver+CUDA (`ubuntu24.04-cuda12` = 570/12.8, `-cuda13.0` = 580/13.0); Docker included | Yes (Stripe) | **Default quotas: H100 = 32 in eu-north1 only, B200 = 0 everywhere** — quota request needed; `shutdown` inside guest auto-reboots (stop via console); egress + public IP free |
| **Hyperstack** | **PCIe $2.50** (spot $2.00), CANADA-1 only; SXM $3.20 but 8x only | B200 $6.00 **8x only** ($48/node); single-GPU Blackwell = **RTX PRO 6000 SE $1.85** (1x/2x/4x/8x) | **Yes, PCIe:** 2x $5.00, 4x $10.00 | Per-minute, prepaid | Cloud-SSD volumes ~$0.07/GB-mo, survive VM deletion, billed while detached; snapshots same rate; public IP ~$5/mo | Full VM, `ubuntu` + sudo, cloud-init; images name the driver: R535/CUDA 12.2, R550/12.4, R570/12.8, R595/13.2, with/without Docker | Yes (Stripe, 3-D Secure) | SHUTOFF VMs bill in full (hibernate instead); spot deleted with no notice; balance at $0 → hibernate → deletion after ~1 week; single H100 in one region |
| **Crusoe Cloud** | **Not offered** (H100 SXM 8x only, $3.90/GPU = $31.20/node) | **No single-GPU Blackwell**; B200 8x / GB200 4x = contact sales | No (2x/4x only for A100/L40S) | Per-second; stopped VM compute not billed | Persistent disk $0.08/GiB-mo, survives VM deletion | Full VM, `ubuntu` + sudo (root login deprecated); driver/CUDA versions not documented, only via `crusoe compute images list` | Yes as of 2026 (non-prepaid card); spot and reservations via sales | Fails the single-GPU requirement outright |
| **CoreWeave** | **Not offered** (HGX H100 8x $49.24/node = $6.16/GPU) | No single GPU; HGX B200 8x $68.80; GB200 4x $42.00; RTX PRO 6000 8x $20.00 | No | Not published (instance-hours) | Object $0.06/GB-mo, distributed FS $0.07; public IP $4/mo | **No VMs** — bare-metal nodes in managed Kubernetes; driver managed per node pool | **No** — org must be approved by Sales | Fails card-self-serve and single-GPU requirements |
| **Together AI (Instant Clusters)** | 8-GPU minimum: H100 $3.99/GPU = **$31.92/node** | B200 $8.19/GPU, 8x only = $65.52/node | No (node = 8) | Hourly | $0.16/GiB-mo with **1 TiB minimum** (~$160/mo) | SSH to Slurm/K8s nodes; driver/CUDA chosen at cluster create | Self-serve for on-demand | 8-GPU floor and storage floor make it ~10x the cost of the alternatives |
| **Modal (serverless)** | H100 SXM **$3.95** ($0.001097/s) + CPU/RAM | **B200 $6.25**; B300 $7.10; RTX PRO 6000 $3.03 | Up to 8 GPUs per container | Per-second, only while container runs; $30/mo free credit, payment method required | Volumes $0.09/GiB-mo (1 TiB/mo free listed) | Root **inside** container (uid 0); `modal shell --gpu H100` gives interactive bash; **not real SSH**, no VM, 24 h max lifetime; custom `nvidia/cuda` images fine; `nvidia-smi` works | Yes | **`gpu="H100"` may be silently upgraded to H200 — use `"H100!"`**; host driver (580.95 / CUDA 13.0) is Modal's, changes without notice |
| **Baseten / Fireworks** | Hosted endpoints only: Baseten H100 $6.50, Fireworks H100 $8.00 | Baseten B200 $9.98, Fireworks B200 $13.00 | n/a | Per-minute / per-second | n/a | **No root shell** (Baseten "Inference SSH" is a non-root `app` user inside a deployed model, enabled by support) | Yes | Not usable for kernel-level determinism work |
| **AWS** | **p5.4xlarge (1x H100 80 GB) $6.88** us-east-1 (Capacity Block $5.19, 1-day minimum, paid up front) | **No single-GPU size**; p6-b200.48xlarge 8x B200 **$113.93/hr** on-demand ($98.84 Capacity Block) | 8x only for H100 SXM | Per-second, 60 s minimum | EBS gp3 $0.08/GB-mo | Full VM, sudo; DLAMI documents driver 595.91.07 / CUDA 12.8-13.2 | Card yes, but **"Running On-Demand P instances" quota defaults to 0 vCPU** — request required, hours to days, sometimes denied for new accounts | Egress $0.09/GB after 100 GB/mo; Capacity Blocks bypass quota but minimum 1 day |
| **GCP** | a3-highgpu-1g (1x H100): **Spot / Flex-start only** — Flex-start **$4.79**, Spot ~$6.57 *(3rd-party)*; on-demand list ~$10.98 *(3rd-party)* but not orderable as on-demand | **No single-GPU size**; a4-highgpu-8g 8x B200 Flex-start **$64.44/hr**, reservation/Spot/Flex-start only | a3-highgpu-2g/4g exist, Spot/Flex-start only ($9.58 / $19.16 Flex-start) | 1-minute minimum then per-second | pd-balanced $0.10, Hyperdisk Balanced $0.08/GiB-mo | Full VM, sudo; DLVM images `*-cu129-ubuntu-2404-nvidia-580` | Card yes, but **GPUS_ALL_REGIONS quota is 0** on new projects; free-trial accounts cannot request GPU quota | Egress $0.12/GiB Premium tier; Spot discount on A3 is "smaller" than usual |
| **Azure** | **NC40ads_H100_v5 (1x H100 NVL 94 GB) $6.98** East US (Spot $1.29) | **No B200 SKU at all**; ND GB200 v6 = 4x GB200 **$108.16/hr** | NC80adis_H100_v5 (2x NVL) $13.96 | Per-minute | Premium SSD v2 ~$0.08/GiB-mo | Full VM, sudo; Ubuntu-HPC image documents driver 580.178.04 / CUDA 13.0.88 | Card yes, but **GPU family quota is 0** and increases are gated on payment history; free accounts cannot get GPU quota | Egress $0.087/GB after 100 GB; community reports of NC/ND quota taking days-weeks or being rejected |

---

## 3. Per-provider notes (what the table cannot hold)

### Lambda Cloud
- Sources: <https://lambda.ai/pricing>, <https://docs.lambda.ai/public-cloud/billing/>, <https://docs.lambda.ai/public-cloud/filesystems/>, <https://lambda.ai/lambda-stack-deep-learning-software>.
- H100 ladder (per GPU-hr): 1x SXM $4.29, 2x $4.19, 4x $4.09, 8x $3.99; 1x PCIe $3.29 (PCIe only 1x or 4x). B200: 1x $6.99, 2x $6.89, 4x $6.79, 8x $6.69. GH200 1x $2.29.
- Billing starts after health checks pass and runs until **terminate**; `sudo shutdown` does not stop billing and there is no stopped state. Postpaid weekly invoices, so no prepaid-balance trap.
- Filesystems: $0.20/GiB-mo, billed hourly on used GiB, must be created in the instance's region and attached at launch. If the 1x H100 and 1x B200 are only available in different regions you need two filesystems (or re-download weights; ingress/egress are free and the NICs are fast).
- Images: Lambda Stack 22.04/24.04 (currently CUDA 12.8.93, PyTorch 2.7.0, NCCL 2.26.2, container toolkit 1.18.1), "GPU Base", plain Ubuntu. Driver version is not published per image — record it with `nvidia-smi` (section 6).
- New accounts have an unpublished instance-count cap that rises as invoices are paid; 1x H100 stock-outs are widely reported.

### Verda (formerly DataCrunch)
- Sources: <https://verda.com/pricing>, <https://docs.verda.com/welcome-to-verda/pricing-and-billing/>, <https://docs.verda.com/cpu-and-gpu-instances/set-up-a-gpu-instance/>, <https://docs.verda.com/storage/deleting-storage/>.
- `datacrunch.io` now redirects to `verda.com`. Instance names encode the SKU: `1H100.80S.30V` (S = SXM), `1B200.180S.30V`.
- Spot is a fixed 50 % discount and can be evicted at any moment; you can set a per-volume policy for what happens to OS/extra volumes on eviction.
- Billing in prepaid 10-minute increments; effective minimum 10 minutes. Card/debit or SEPA; no sales.
- **Gotcha:** a *shut-down* instance keeps billing — hibernate or delete. If the prepaid balance reaches zero, instances stop and volumes are deleted after 96 h.
- Regions: Finland and Iceland only. Irrelevant for SSH work; note it in the envelope.
- A Verda docs example shows driver 580.126.09 on Blackwell.

### RunPod
- Sources: <https://www.runpod.io/pricing>, <https://docs.runpod.io/pods/pricing>, <https://docs.runpod.io/pods/storage/types>, <https://docs.runpod.io/pods/configuration/use-ssh>, <https://docs.runpod.io/references/gpu-types>.
- A pod is a Docker container: you are root inside it, `nvidia-smi` works, but you cannot run Docker inside. Pick any public image (`nvcr.io/nvidia/pytorch:…`, `vllm/vllm-openai:…`) or a RunPod template.
- Network Volumes are the only storage that survives *terminate*; they live in one datacenter and pods must be started there, which constrains which GPUs are in stock. Community Cloud has no network volumes.
- Deploy UI → Additional filters → **CUDA Versions** (API `allowedCudaVersions`) lets you pin the host CUDA driver level; pods expose `CUDA_VERSION`. Driver still varies host to host, so record it every run.
- Must hold ≥ 1 hour of credit for the chosen config to deploy; default account spend cap $80/hr; KYC only for crypto payments.
- Interruptible ("spot") pods exist via API (`interruptible: true`, `bidPerGpu`) with no documented notice period.

### Vast.ai
- Sources: <https://docs.vast.ai/guides/reference/billing>, <https://docs.vast.ai/documentation/reference/faq/rental-types>, <https://docs.vast.ai/guides/instances/storage/types>, live offers from `console.vast.ai/api/v0/bundles/` (2026-09-09 17:18 UTC).
- Per-second billing, no minimum; storage billed per second whether the instance is running or stopped; bandwidth billed per byte at host-set rates.
- Each offer exposes `driver_version` and `cuda_max_good`, filterable in UI/CLI (e.g. `cuda_max_good>=12.8`), which is nice for the envelope — but hosts are heterogeneous consumer/prosumer machines, "Volumes" cannot leave the physical host, and unverified hosts can see your data. Use the "Secure Cloud (trusted datacenters)" filter for anything sensitive.
- Interruptible instances are stopped (processes killed) whenever outbid; on-demand always outranks bids.
- Single-B200 inventory was three machines on the day of research; treat B200 on Vast as opportunistic.

### Nebius
- Sources: <https://nebius.com/prices>, <https://docs.nebius.com/compute/resources/pricing>, <https://docs.nebius.com/compute/resources/quotas-limits>, <https://docs.nebius.com/compute/storage/boot-disk-images>.
- Clean VM product with per-second billing and documented driver/CUDA per image family, but presets are 1-GPU or 8-GPU only (no 2x/4x H100), and **default B200 quota is 0** (H100 default 32, only in eu-north1). Quota requests are a form, not a sales call, but add latency.
- Do not `shutdown` from inside the guest — Nebius treats it as a failure and reboots the VM (billing continues); stop from console/CLI. Stopped VMs are not billed for compute.
- Preemptible VMs get SIGTERM 60 s before stop. Egress and public IPs are free.

### Hyperstack
- Sources: <https://www.hyperstack.cloud/gpu-pricing>, <https://docs.hyperstack.cloud/docs/hardware/flavors>, <https://docs.hyperstack.cloud/docs/billing/states-and-billing>, <https://docs.hyperstack.cloud/docs/virtual-machines/images>.
- Cheapest single H100 among the "real VM" providers ($2.50 PCIe), and the only one with **2x/4x H100 nodes at PCIe prices** ($5/$10 per hr) — but all H100 PCIe flavours live in CANADA-1, and SXM/B200 are 8x-only.
- Image catalogue names the driver branch and CUDA (R535/12.2 … R595/13.2), which is good for the envelope.
- SHUTOFF VMs are billed at full rate; use HIBERNATE (bills root disk + IP only). Spot VMs are deleted without notice.
- Single-GPU Blackwell here is RTX PRO 6000 Server Edition ($1.85) — an SM120 part, see section 4.

### Crusoe, CoreWeave, Together, Modal, Baseten, Fireworks
- **Crusoe** is now card self-serve (non-prepaid card, $5 credit) but sells H100 only as 8x nodes ($31.20/hr) and Blackwell only via sales; single-GPU flavours are L40S/A100. Persistent disks $0.08/GiB-mo. Fails the single-H100 requirement.
- **CoreWeave** still requires Sales approval to activate an org; instances are bare-metal nodes in managed Kubernetes (no VMs, drivers managed per node pool, 8x minimum). Not viable for a solo card-paying user.
- **Together AI** Instant Clusters are self-serve but 8-GPU minimum ($31.92/hr H100, $65.52/hr B200), hourly billing, and storage has a 1 TiB floor (~$160/mo).
- **Modal** is the best of the serverless options if you can live without real SSH: root inside a `nvidia/cuda` container, `modal shell --gpu B200 --image … --volume …` gives an interactive bash, `nvidia-smi` works, Volumes $0.09/GiB-mo, $30/mo free credit, academic credits up to $10k. Pin GPUs with `gpu="H100!"` (plain `"H100"` may be upgraded to H200; `"B200+"` may become B300). The host driver is Modal's and can change, and containers live at most 24 h.
- **Baseten / Fireworks** are hosted-inference products; no root shell. Fireworks raised prices on 2026-09-01 (H100 $8/hr, B200 $13/hr).

### Big three (AWS, GCP, Azure)
- Sources: AWS on-demand price feed for us-east-1, <https://aws.amazon.com/ec2/capacityblocks/pricing/>, <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-on-demand-instances.html>; <https://docs.cloud.google.com/compute/docs/gpus>, <https://cloud.google.com/products/dws/pricing>; Azure Retail Prices API (East US), <https://learn.microsoft.com/en-us/azure/virtual-machines/quotas>.
- **Single Blackwell GPU is not sold by any of them.** Smallest units: Azure ND GB200 v6 (4x GB200, $108.16/hr), GCP a4-highgpu-8g (8x B200, Flex-start $64.44/hr, no plain on-demand), AWS p6-b200.48xlarge (8x B200, $113.93/hr on-demand or $98.84/hr Capacity Block with a 1-day minimum paid up front).
- Single H100: AWS p5.4xlarge $6.88/hr (per-second, 60 s min), Azure NC40ads_H100_v5 $6.98/hr (H100 **NVL 94 GB**, a different SKU from the 80 GB SXM/PCIe parts — matters for the envelope), GCP a3-highgpu-1g only as Spot (~$6.57) or Flex-start ($4.79).
- **Quota friction:** all three start new accounts at **0 GPU quota** (AWS "Running On-Demand P instances" 0 vCPU; GCP `GPUS_ALL_REGIONS` 0; Azure NC/ND family 0). Requests are self-service forms, not sales calls, but approval is undocumented (community: hours to days on AWS/GCP, days to weeks and frequent rejections on fresh Azure PAYG subscriptions). Free-trial tiers on GCP/Azure cannot hold GPU quota at all. AWS Capacity Blocks bypass the On-Demand quota but have a 1-day minimum.
- Egress is the hidden cost: $0.09/GB (AWS), $0.12/GiB (GCP Premium), $0.087/GB (Azure) after 100 GB/mo. Pulling 50 GB of weights *in* is free; pushing large trace dumps *out* is not.
- Images do document driver/CUDA precisely (AWS DLAMI 2026-08-28: driver 595.91.07, CUDA 12.8-13.2; GCP DLVM `nvidia-580`/CUDA 12.9; Azure Ubuntu-HPC 2404: 580.178.04 / CUDA 13.0.88), which is the one thing they do better than the neoclouds.

---

## 4. Which Blackwell? B200 (SM100) vs RTX PRO 6000 / RTX 5090 (SM120)

The cheap Blackwell options (RunPod RTX PRO 6000 $2.09, RTX 5090 $0.99; Nebius RTX PRO 6000 $1.80; Hyperstack $1.85; Vast from $1.33) are **SM120** parts. B200/B300/GB200 are **SM100**. nvfp4 runs on both, but through different kernels:

- vLLM ships separate CUTLASS NVFP4 kernels for SM120 (`nvfp4_scaled_mm_sm120_kernels.cu`, `ENABLE_NVFP4_SM120`, present since v0.13) — <https://github.com/vllm-project/vllm/pull/21309>, <https://github.com/vllm-project/vllm/pull/24968>.
- Open correctness/performance issues on SM120 as of this research: vLLM 0.24 routes ModelOpt *mixed* NVFP4 checkpoints on RTX 5090 to the Marlin W4A16 fallback with a misleading "no native FP4" warning (open, 2026-07-06, <https://github.com/vllm-project/vllm/issues/47749>); SGLang produced NaN/garbage on RTX PRO 6000 with `--fp4-gemm-backend flashinfer_cutlass` (reported 2026-02-18, <https://github.com/sgl-project/sglang/issues/18954>); SM120 NVFP4 MoE kernels were still a feature request (<https://github.com/vllm-project/vllm/issues/31085>).
- SGLang's documented nvfp4 path (`--quantization modelopt_fp4`) targets SM100 first — <https://docs.sglang.io/docs/advanced_features/quantization>.

For a *determinism* study this means SM120 results are a different kernel family from B200 results, not a cheaper proxy for them. Recommendation: **B200 is the reference Blackwell target**; an RTX PRO 6000 session (~$2/hr) is an optional extra data point ("does the same claim hold on SM120?") and a cheap place to debug the nvfp4 loading path before paying B200 rates. If a claim only needs "any Blackwell with native FP4 tensor cores," the cheaper card is fine; if the claim is about production B200 serving, do not substitute.

---

## 5. Cost estimate for the validation campaign

Assumptions: 20 GPU-hours on 1x H100, 10 GPU-hours on 1x B200, one 50 GB persistent volume for 2 months. Compute-only figures ignore the ~10-20 % of wall-clock spent on instance boot, image pull and debugging, so a realistic budget is the last column.

| Plan | H100 (20 h) | B200 (10 h) | Storage (50 GB × 2 mo) | Compute + storage | Realistic budget (+25 %) |
|---|---|---|---|---|---|
| **Lambda, H100 PCIe** | 20 × $3.29 = $65.80 | 10 × $6.99 = $69.90 | 50 × $0.20 × 2 = $20 | **$155.70** | ~$195 |
| **Lambda, H100 SXM** | 20 × $4.29 = $85.80 | $69.90 | $20 | **$175.70** | ~$220 |
| Lambda, two regions (2 filesystems) | as above | as above | $40 | $175.70-195.70 | ~$245 |
| **Verda, on-demand** | 20 × $3.25 = $65.00 | 10 × $6.11 = $61.10 | 50 × $0.20 × 2 = $20 | **$146.10** | ~$185 |
| Verda, all spot | 20 × $1.63 = $32.60 | 10 × $3.06 = $30.60 | $20 | $83.20 | ~$105 (plus eviction risk) |
| **RunPod Secure** (SXM) | 20 × $3.49 = $69.80 | 10 × $6.79 = $67.90 | 50 × $0.07 × 2 = $7 | **$144.70** | ~$180 |
| RunPod Secure (PCIe) | 20 × $2.89 = $57.80 | $67.90 | $7 | $132.70 | ~$165 |
| Nebius | 20 × $3.85 = $77.00 | 10 × $7.15 = $71.50 | 50 × $0.071 × 2 = $7 | $155.50 | ~$195 (after B200 quota approval) |
| Hyperstack H100 + RunPod B200 | 20 × $2.50 = $50.00 | $67.90 | $7 + $7 | $131.90 | ~$165 (two providers, two envelopes) |
| AWS, for comparison | 20 × $6.88 = $137.60 | 10 × $113.93 (8x node) = $1,139 | EBS 50 × $0.08 × 2 = $8 | $1,285 | + quota request |

Later tensor-parallel phase, 4x H100 in one node, per hour: Hyperstack PCIe $10.00, Verda SXM $13.00, RunPod Secure SXM ~$13.96, Lambda SXM $16.36, Nebius/Crusoe/CoreWeave 8x-only ($26-$49). A 10-hour 4x SXM TP campaign is therefore ~$130-165 on Verda/RunPod/Lambda.

**Bottom line:** the whole H100 + B200 validation campaign is a **$150-$250** item on any of the three recommended providers. The price spread between them (~$30) is smaller than the cost of one wasted debugging afternoon, so choose on environment fidelity (real VM, documented image, same provider for both GPUs) rather than on the hourly rate.

---

## 6. Recording the hardware/software envelope reproducibly

Run this at the start of every session and commit the output next to the run's results. It works on every provider above (VM or container); on container providers (RunPod, Vast, Modal) the driver line is the *host's* and can differ between pods, which is exactly why it must be recorded per run.

```bash
# envelope.sh — capture GPU SKU, driver, CUDA, framework versions
set -euo pipefail
echo "provider=${PROVIDER:-unknown} instance_type=${INSTANCE_TYPE:-unknown} region=${REGION:-unknown}"
nvidia-smi --query-gpu=name,pci.device_id,pci.sub_device_id,vbios_version,driver_version,memory.total,compute_cap,clocks.max.sm,clocks.max.mem,power.limit --format=csv
nvidia-smi -q | grep -E "Product Brand|Product Architecture|CUDA Version|Persistence Mode|MIG Mode" | sort -u
cat /proc/driver/nvidia/version 2>/dev/null || true
command -v nvcc >/dev/null && nvcc --version | tail -1
python - <<'PY'
import torch, platform
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("device", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
try:
    import vllm; print("vllm", vllm.__version__)
except Exception: pass
try:
    import sglang; print("sglang", sglang.__version__)
except Exception: pass
print("python", platform.python_version(), platform.platform())
PY
uname -r; cat /etc/os-release | grep PRETTY_NAME
```

Notes for a tight envelope:

- **Pin the container image by digest** (`nvcr.io/nvidia/pytorch:25.08-py3@sha256:…`, `vllm/vllm-openai:v0.x.y@sha256:…`) rather than by tag, and run the same image on Lambda's VM, Verda's VM and RunPod's pod. The driver is then the only host-supplied component, and the script above records it.
- **SXM vs PCIe vs NVL are different SKUs** (different memory bandwidth, TDP and boost clocks; Azure's NC H100 v5 is the 94 GB NVL part). `pci.device_id` disambiguates: H100 SXM5 80 GB = `0x2330`, H100 PCIe 80 GB = `0x2331`, H100 NVL = `0x2321`, B200 = `0x2901`, RTX PRO 6000 Blackwell Workstation Edition = `0x2BB1`, GeForce RTX 5090 = `0x2B85` (IDs from NVIDIA driver README, supported-chips list). Record it; do not rely on the provider's marketing name.
- Provider-side pins: Lambda Stack version page, Verda image name (`…-cuda-12.8-open-docker`), Nebius image family (`ubuntu24.04-cuda12` = driver 570 / CUDA 12.8), Hyperstack image name (R570/CUDA 12.8), RunPod "CUDA Versions" filter / `CUDA_VERSION` env var, Vast `driver_version` / `cuda_max_good` offer fields, Modal docs (host 580.95.05 / CUDA 13.0 at research time).
- Also record `NCCL_*`, `CUBLAS_WORKSPACE_CONFIG`, `torch.use_deterministic_algorithms`, vLLM's attention backend and `--enforce-eager`, and the exact nvfp4 checkpoint hash — these move results more than the cloud does.

---

## 7. Practical launch checklist (Lambda primary)

1. Sign up with a credit card (expect a $10 pre-authorisation). Check the instance-count cap on a new account; the first invoice raises it.
2. Create a **Filesystem (50-100 GB)** in the region where 1x H100 *and* 1x B200 both show availability; if no single region has both, accept two filesystems (~$10/mo each) or re-download weights (no ingress fee).
3. Launch `gpu_1x_h100_pcie` (or `_sxm5`) with the Lambda Stack 24.04 image, filesystem attached. Run `envelope.sh`. Pull the pinned vLLM/SGLang image by digest with the preinstalled Docker + NVIDIA Container Toolkit.
4. Run the validation set; copy results to the filesystem; **terminate** (there is no stop).
5. Repeat on `gpu_1x_b200` for the nvfp4 runs. Same image digest, same script.
6. If Lambda shows no 1x B200 stock: Verda `1B200.180S.30V` (root SSH, Docker preinstalled, $6.11/hr, 10-min increments; delete or hibernate afterwards, do not just shut down). If neither has stock: RunPod Secure Cloud B200 pod with a network volume in that datacenter and the CUDA-version filter set.
7. Optional cheap pre-flight for the nvfp4 loading path: RunPod RTX PRO 6000 Blackwell ($2.09/hr) — but label those results SM120, not B200.
8. TP phase: Lambda `gpu_4x_h100_sxm5` ($16.36/hr) or Verda 4x H100 ($13.00/hr); both NVLink.

---

## Sources

Official pages and docs consulted on 2026-09-09 (all prices above trace to one of these unless marked 3rd-party):

- Lambda: <https://lambda.ai/pricing>, <https://docs.lambda.ai/public-cloud/billing/>, <https://docs.lambda.ai/public-cloud/filesystems/>, <https://docs.lambda.ai/public-cloud/on-demand/>, <https://lambda.ai/lambda-stack-deep-learning-software>
- Verda/DataCrunch: <https://verda.com/pricing>, <https://docs.verda.com/welcome-to-verda/pricing-and-billing/>, <https://docs.verda.com/cpu-and-gpu-instances/set-up-a-gpu-instance/>, <https://docs.verda.com/cpu-and-gpu-instances/shutdown-hibernate-and-delete>, <https://docs.verda.com/storage/deleting-storage/>
- RunPod: <https://www.runpod.io/pricing>, <https://docs.runpod.io/pods/pricing>, <https://docs.runpod.io/accounts-billing/billing>, <https://docs.runpod.io/pods/storage/types>, <https://docs.runpod.io/pods/configuration/use-ssh>, <https://docs.runpod.io/references/gpu-types>
- Vast.ai: <https://docs.vast.ai/guides/reference/billing>, <https://docs.vast.ai/documentation/reference/faq/rental-types>, <https://docs.vast.ai/guides/instances/storage/types>, <https://docs.vast.ai/host/verification-stages>, live offers API `https://console.vast.ai/api/v0/bundles/`
- Nebius: <https://nebius.com/prices>, <https://docs.nebius.com/compute/resources/pricing>, <https://docs.nebius.com/compute/resources/quotas-limits>, <https://docs.nebius.com/compute/storage/boot-disk-images>, <https://docs.nebius.com/signup-billing/sign-up>
- CoreWeave: <https://www.coreweave.com/pricing>, <https://docs.coreweave.com/get-started>, <https://docs.coreweave.com/platform/instances/gpu-instances>
- Hyperstack: <https://www.hyperstack.cloud/gpu-pricing>, <https://docs.hyperstack.cloud/docs/hardware/flavors>, <https://docs.hyperstack.cloud/docs/billing/states-and-billing>, <https://docs.hyperstack.cloud/docs/virtual-machines/images>
- Crusoe: <https://www.crusoe.ai/cloud/pricing>, <https://docs.crusoecloud.com/compute/virtual-machines/overview>, <https://docs.crusoecloud.com/create-an-account/>, <https://docs.crusoecloud.com/storage/disks/overview>
- Together: <https://www.together.ai/pricing>, <https://docs.together.ai/docs/instant-clusters>
- Modal: <https://modal.com/pricing>, <https://modal.com/docs/guide/gpu>, <https://modal.com/docs/guide/cuda>, <https://modal.com/docs/reference/cli/shell>, <https://modal.com/docs/guide/volumes>
- Baseten: <https://www.baseten.co/pricing/>, <https://docs.baseten.co/inference/ssh>; Fireworks: <https://fireworks.ai/pricing>
- AWS: <https://aws.amazon.com/ec2/instance-types/p5/>, <https://aws.amazon.com/ec2/instance-types/p6/>, <https://aws.amazon.com/ec2/capacityblocks/pricing/>, <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-on-demand-instances.html>, <https://aws.amazon.com/ebs/pricing/>, DLAMI release notes 2026-09-02
- GCP: <https://docs.cloud.google.com/compute/docs/gpus>, <https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-accelerator-optimized>, <https://cloud.google.com/products/dws/pricing>, <https://cloud.google.com/compute/disks-image-pricing>, <https://cloud.google.com/vpc/network-pricing>; on-demand/Spot A3 list prices 3rd-party (DevZero, Thunder Compute)
- Azure: Retail Prices API `prices.azure.com/api/retail/prices` (East US), <https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ncadsh100v5-series>, <https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nd-gb200-v6-series>, <https://learn.microsoft.com/en-us/azure/virtual-machines/quotas>, <https://azure.microsoft.com/en-us/pricing/details/bandwidth/>
- nvfp4 / SM120: <https://github.com/vllm-project/vllm/pull/21309>, <https://github.com/vllm-project/vllm/issues/47749>, <https://github.com/vllm-project/vllm/issues/31085>, <https://github.com/sgl-project/sglang/issues/18954>, <https://docs.sglang.io/docs/advanced_features/quantization>
