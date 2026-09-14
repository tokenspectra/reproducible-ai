# reproducible-ai: bitwise-deterministic, replayable LLM inference

Status: design accepted 2026-09-09, no code yet. Decisions were reached in a
structured interview; the question-by-question log is in the appendix and the
hard-to-reverse ones are recorded as ADRs under `docs/adr/`. Vocabulary is in
`/CONTEXT.md`. Supporting research is under `docs/research/`.

## 1. Goal

Build an inference system, and the artifact format around it, such that:

1. A request produces **bitwise-identical logits at every step** on a fixed
   hardware/software envelope, regardless of batch neighbors, batch size,
   process restarts, chunked-prefill boundaries, or whether a position was
   reached by prefill or by token-by-token decode.
2. Every generation is written to a **trace** that carries the inputs, the
   seeds, the envelope, and per-step evidence, and is self-contained.
3. A **verifier** that has only the trace and the whitelisted weights can
   re-execute it on independent hardware of the same envelope and either
   confirm every step or name the first diverging step.
4. A trace can be **resumed**: replayed without re-verifying and then continued
   with a new message, both by fast prefill of the recorded prefix and by exact
   step-by-step replay, with both paths provably reaching the same state.

The determinism claim is deliberately scoped (ADR-0001): same GPU architecture,
same compiled kernels, same dtype config, same weights. Cross-architecture and
cross-vendor bitwise identity are non-goals; a replay on a different envelope is
refused unless explicitly allowed, and then reported as best effort.

## 2. Why: the verification-plan requirements

The project is motivated by the technical requirements in the ai-2040
verification plan (`docs/research/verification-plan-requirements.md`). The
parts that shape this design:

- The verifier is **adversarial** and runs on its own hardware. Nothing the
  operator computes (a hash, a signature) is evidence; only independent
  re-execution is. Hashes in the trace are diagnostics and integrity checks,
  not proof.
- The unit of verification is a **packet**: inputs, instructions, seed,
  outputs, self-contained and replayable in isolation. Granularity ranges from
  one forward pass to many GPU-hours, so the schema must not hard-code one.
- **Seeds may be supplied by the auditor.** The sampler must be a pure function
  of (logits, seed).
- **Cross-machine reproduction** on standardized hardware is the implicit bar.
  The envelope must be specified precisely enough to serve as a standard.
- Periodic memory wipes mean a packet cannot depend on long-lived KV or prefix
  state the verifier cannot rebuild. Hence prefill-of-prefix must equal
  decode-of-prefix bitwise.
- Intra-node multi-GPU is in scope; cross-node scale-out is not. Overhead must
  be small; the verifier's recompute budget is about 1%.
- Agentic, multi-turn use is in scope, which is why the trace is a session.

## 3. Prior art and the gap

Full notes: `docs/research/local-gpu-feasibility.md`,
`docs/research/sglang-deterministic-audit.md`.

**Exists and works.** Thinking Machines' `batch_invariant_ops` (MIT, Triton:
matmul, log_softmax, mean). vLLM's batch-invariant mode (CUDA-only GEMM
overrides, sampler unchanged). SGLang's `--enable-deterministic-inference`:
batch-invariant GEMM/RMSNorm/softmax, fixed-split attention on FlashInfer, FA3
and Triton, seeded Gumbel sampling whose noise is a counter hash of
(seed, absolute token position, vocab index), fixed MoE config, pinned
all-reduce on NVIDIA and AMD. Verified locally on the RX 6700 XT: a fixed-tile
Triton GEMM is bitwise batch-invariant while `torch.mm` via hipBLAS is not
(fp16, K=11008: row 0 differs across M in {1, 7, 64, 512, 1024}).

**Does not exist.** Neither engine records an envelope (no weight hash, git SHA,
kernel versions, GPU arch anywhere in a request, response or endpoint). Neither
has a verifier or a per-step evidence format; SGLang exposes top-k logprobs but
no full-logits option. Neither makes or tests any cross-machine claim; every
test is batch-composition invariance on one server. SGLang's top-p path uses an
unstable GPU sort with Gumbel noise indexed by sorted column, so exact
probability ties are not batch-invariant. SGLang's tracking issue lists as open:
prefill-with-radix-cache parity, blockwise and per-channel FP8 GEMM, FP8 MoE,
NVFP4 GEMM/MoE, expert parallelism, DP attention, Blackwell TP4.

**Therefore** the contribution of this project is: the trace/packet format, the
verifier, the empirical characterization of what the envelope must contain, the
prefill-vs-decode and resume guarantees, stochastic-rounding and quantized-format
determinism, and a batch-invariant RNG contract that is engine-neutral. The
batch-invariant bf16 kernel layer is prior art that we test against and reuse
(ADR-0002).

## 4. Architecture

Two personas, one shared library, two engines.

```
                 +----------------------------+
                 |  reproducible_ai (library) |
                 |  trace schema, hashing,    |
                 |  envelope capture, RNG,    |
                 |  verify logic, test matrix |
                 +------+--------------+------+
                        |              |
          +-------------+--+       +---+----------------+
          | reference oracle |     | SGLang adapter     |
          | plain torch,     |     | (cloud, P3)        |
          | local, tiny      |     | trace writer +     |
          +------------------+     | verifier backend   |
                                   +--------------------+
   record  <-- operator side: runs requests, writes traces
   verify  <-- verifier side: trace + weights in, report out
```

- **`record`** (operator side): serves or batch-runs requests through an engine
  backend and writes traces.
- **`verify`** (verifier side): takes a trace file and a weights path, rebuilds
  and checks the envelope, replays, compares per-step evidence, reports. It is
  forbidden by construction from reading anything other than the trace, the
  weights and the environment it runs in (ADR-0006).
- **Reference oracle**: a minimal plain-torch engine for Qwen3-0.6B and
  SmolLM2. Contiguous KV in v0, paged KV afterwards. Seeded sampler. In-process
  continuous-batching scheduler so the chaos batcher is realistic. Exists to
  make the trace format and the verifier testable end to end on the local GPU;
  it is not a serving engine and has no HTTP layer.
- **Kernel lab**: the custom kernels this project actually needs, all Triton:
  the GEMM epilogue with seeded stochastic rounding, FP8 and NVFP4 dynamic
  quantizers (emulated locally, real on cloud), the counter-based RNG, a
  stable-tie-break sampler, and any op that a contract test shows torch does
  not keep invariant. Batch-invariant bf16 GEMM/RMSNorm come from Thinking
  Machines' kernels where needed locally; ours only where theirs are absent.
- **SGLang adapter** (P3): plugs the trace writer into SGLang's deterministic
  mode and adds a verifier backend that drives SGLang. vLLM is a second adapter
  later; two adapters are the evidence that the format is engine-neutral.
- **Test harness**: the seven-test determinism matrix (section 10), runnable
  locally for tests 1-4 and 6, on cloud for 5 and 7.

Offline Python API plus CLI only in v1. Messages are system, user and assistant;
tool-call turns are a reserved schema slot.

## 5. Trace format

A trace is a **session**: a header followed by an ordered stream of events, one
JSON object per line (`.trace.jsonl`). Each event carries the BLAKE3 hash of
the previous event, so a trace is append-only and tamper-evident (integrity,
not proof). Large debug payloads live in a safetensors sidecar referenced by
step records.

### 5.1 Header (line 1)

```json
{
  "kind": "header",
  "format_version": 1,
  "trace_id": "<uuid>",
  "created_at": "<iso8601>",
  "root_seed": "<64 hex chars>",
  "seed_source": "os_random | external",
  "model": {"whitelist_id": "Qwen/Qwen3-0.6B", "weights_file_hash": "...",
            "weights_memory_hash": "...", "tokenizer_hash": "...",
            "chat_template_hash": "..."},
  "dtype_config": {"weights": "bf16", "activations": "bf16", "residual": "bf16",
                   "accumulate": "fp32", "kv_cache": "bf16",
                   "output_cast": "rne | stochastic", "quant": null},
  "envelope": {"hard": {...}, "soft": {...}},
  "manifest": {"engine": "reference | sglang", "engine_git_sha": "...",
               "dirty": false, "lockfile_hash": "...", "image_digest": null},
  "prev_hash": null,
  "hash": "..."
}
```

### 5.2 Events

- `turn_start`: `turn_index`, `role`, `messages_delta` (the new message(s) as
  given), `rendered_token_ids` (the tokens actually fed, so the verifier can
  re-render and assert equality), and for assistant turns the `turn_seed`,
  derived `sampling_seed` and `sr_seed`, and `sampling_params` (temperature,
  top_k, top_p, min_p, max_tokens, stop).
- `step`: `turn_index`, `position` (absolute token position in the sequence),
  `token_id`, `logits_hash` (BLAKE3-128 of the fp32 pre-temperature logits in
  canonical little-endian byte order, ADR-0009), optional `layer_hashes` and
  `sidecar_ref` at debug level.
- `turn_end`: `finish_reason`, `num_steps`, `turn_hash` (hash over the turn's
  events), `wall_time_ms` (informational, never compared).

Every event has `prev_hash` and `hash`. A step range plus the header plus the
prefix tokens is a packet; a `packet` extraction command is deferred, but the
schema guarantees it is derivable.

### 5.3 Envelope: hard and soft

Soft where possible (ADR-0005). A **hard** field mismatch makes `verify`
refuse (overridable with `--allow-mismatch`, which downgrades the result to
best-effort and still reports the first divergence). A **soft** field is
recorded and compared but never blocks. A soft field is promoted to hard the
moment a test shows it can flip a bit, and the promotion is recorded as a
finding.

Hard (six fields): `weights_memory_hash`, `tokenizer_hash` +
`chat_template_hash`, `dtype_config`, `kernel_registry_hash` (hash of the
**compiled device binaries** of every custom kernel, which subsumes compiler
version, flags and target), `gpu_arch` (compute capability or gfx target, never
the SKU string), `parallel_config` (world size, layout, collective algorithm).

Soft: torch version, Triton version, BLAS library versions, driver, runtime,
GPU SKU string and PCI device id, VRAM, SM count, hostname, engine git SHA and
dirty flag, timestamps. Expected first promotion: torch version, because
elementwise ops stay in torch.

### 5.4 Manifest

The "instructions" of a packet. In v1 it is the recorded facts plus the uv
lockfile hash. From the cloud phase it gains a container `image_digest`, and
the verifier runs inside that image. The manifest is embedded, never
referenced by URL, so a trace stays self-contained.

## 6. Seeds and the RNG contract

- One 256-bit **root seed** per session, from OS randomness by default or
  supplied externally (`--seed`); `seed_source` records which. The engine
  behaves identically either way.
- **Turn seed** = KDF(root_seed, turn_index); overridable per turn. Derived
  **sampling seed** = KDF(turn_seed, "sample") and **SR seed** =
  KDF(turn_seed, "sr") are recorded explicitly in `turn_start` and each is
  independently overridable, so "same rounding mask, different sample"
  experiments are one flag.
- Generator: **Philox4x32-10** with an explicit key/counter layout, implemented
  identically in Triton and in a CPU reference (numpy) so masks and draws are
  testable bit-for-bit.
- **Element identity is batch-invariant** (ADR-0004). Counters never use the
  position of an element in a batched tensor. Stochastic rounding uses
  (sr_seed, tensor_id, position, feature_index); sampler draws use
  (sampling_seed, position, draw_index). `position` is the absolute token
  position in the sequence, which makes prefill and decode produce the same
  mask and the same draw for the same token, and replaces any notion of
  "timestep". `tensor_id` is a stable identifier from the model graph (layer
  index, op name, output slot), never a pointer or allocation order.
- Sampling is a pure function of (fp32 logits, sampling_seed, position,
  sampling_params). Argmax and all top-k/top-p cutoffs break ties by lowest
  token id, using a stable ordering, so exact ties are batch-invariant.

## 7. Resume

Two modes, both first-class (Q16):

- **Fast resume**: prefill the recorded prefix in one shot (chunked or not) and
  continue. This is what an operator or verifier uses in practice and what the
  verification plan's memory wipes require.
- **Exact replay**: re-execute the recorded decode steps one token at a time,
  reproducing the original execution path.

Exact replay is the oracle: test 6 asserts that both reach bitwise-identical
KV state and next-step logits. `verify` uses fast resume by default and falls
back to exact replay to localize a divergence. `--no-verify` replays a trace
without comparing evidence (still deterministically) and then accepts a new
user message, appending a new turn to the same trace.

## 8. Kernel rules

Every custom kernel in the registry obeys:

1. **Launch geometry and reduction order are functions of tensor shapes and
   the dtype config only.** Never of device properties (SM count, occupancy,
   clocks) and never of batch composition. No persistent "one program per SM"
   grids, no occupancy-driven split-KV, no timing-based autotuning
   (ADR-0007). This is what lets H100 PCIe and SXM share an envelope.
2. GEMM: tile config chosen from (N, K, dtype) only; no split-K, or a fixed
   split-K independent of M; fp32 accumulation; the SR epilogue draws its mask
   from request-local coordinates.
3. Attention: fixed KV-block iteration order; decode split-KV of fixed size
   per KV length, combined in a fixed order.
4. No atomics on floating-point outputs. No data-dependent early exits that
   change reduction order.
5. Every kernel has a stable `kernel_id`, its compiled binary is hashed into
   the registry hash, and it ships with a batch-invariance test (same row,
   M in {1, 7, 64, 512, 1024}) and a CPU reference where feasible.
6. Elementwise ops (SiLU-gate, RoPE, residual add, embedding) may stay in torch
   under an explicit fp32-math-then-cast contract with a per-op numerical test.
   An op moves to a custom kernel only when its test fails or fusion is needed.

Local constraint: RDNA2 has 64 KB LDS, wave32 and no matrix cores, so tiles are
small (64x64x32, 4 warps, `num_stages <= 2`). Torch 2.9.1 + Triton 3.5.1 is the
local pin; Triton 3.6 through 3.8 miscompile bf16 elementwise on gfx10.3
(ADR-0008).

## 9. Precision contract and number formats

Default dtype config: bf16 weights, activations and residual stream; fp32 GEMM
accumulation; fp32 math inside RMSNorm, RoPE, attention softmax and the
sampler; fp32 logits hashed before any cast. This matches HF and vLLM so the
tolerance oracle (HF transformers, not bitwise) is meaningful. fp16 is an
explicitly recorded alternative config for local speed (RDNA2 does fp16 about
three times faster than bf16).

Output cast: round-to-nearest-even by default; **stochastic rounding** as an
opt-in `output_cast`, applied inside the GEMM epilogue using the SR RNG
contract. SR is the demonstration that seeded randomness inside kernels can be
batch-invariant, and it is what the quantized tiers need.

Quantized tiers, following what serving fleets actually run:

| Tier | Where | Format | Notes |
|---|---|---|---|
| 1 | local + cloud | bf16 / fp16, fp32 acc | v1 |
| 2 | H100 | FP8 W8A8 e4m3, per-token dynamic activation scale, fp32 acc; FP8 KV cache | quantizer amax is a fixed-order per-row reduction; SR opt-in at the cast |
| 3 | B200 (SM100) | NVFP4 W4A4: e2m1 values, e4m3 block scales over 16, fp32 tensor scale; FP8 KV | SM120 cards run different kernels and are never a determinism claim |

Weight-only int4 is out of scope: its arithmetic is bf16 and it teaches nothing
about determinism. Locally, tiers 2 and 3 are emulated (quantize/dequantize in
software, compute in bf16) to test the quantizer and the SR mask bit-for-bit.

## 10. Determinism test matrix

Pass criterion for all: identical per-step `logits_hash` (and therefore token
ids) across every condition.

1. Same request repeated N times, fresh process each time.
2. **Chaos batcher**: the request batched with randomly generated neighbors of
   random lengths at batch sizes 1 through B, random arrival order, random
   preemption.
3. Chunked prefill with varied chunk boundaries.
4. Exact replay vs fast resume (section 7).
5. Two different cloud instances of the same GPU SKU and image.
6. Position k reached via prefill of k tokens vs k single-token decode steps.
7. Same trace on H100 SXM and H100 PCIe (same compute capability, different
   SM count).

Local covers 1-4 and 6. Cloud covers 5 and 7. Every promotion of a soft
envelope field must cite the test that forced it.

## 11. `verify` semantics

Input: trace path, weights path. Steps: parse and check the hash chain;
capture the local envelope and compare (refuse on hard mismatch unless
`--allow-mismatch`); re-render each turn's messages and assert the rendered
token ids match; replay (fast resume per turn, exact replay on divergence to
localize); compare every step's `logits_hash` and `token_id`.

Output: a JSON report (`status: pass | envelope_mismatch | divergence |
render_mismatch | chain_broken`, mismatched fields, first diverging turn and
position, expected vs observed hashes, and with a debug sidecar the first
diverging layer) plus a one-screen human summary. Exit code 0 only on `pass`.
`--steps a:b` verifies a range (the packet case). `--no-verify` is the resume
path described in section 7.

## 12. Models

- Dense, local and cloud: **Qwen3-0.6B** (primary: GQA, QK-norm, RMSNorm,
  SwiGLU, RoPE, Apache-2.0) and **SmolLM2** (secondary, Llama-style).
- MoE, cloud only, P4: **Qwen3-30B-A3B** (official FP8 checkpoint, community
  NVFP4 checkpoints). MoE-specific problems to plan for: top-k routing ties,
  grouped GEMM whose tiling depends on per-expert token counts, expert-parallel
  all-to-all.

## 13. Environments

**Local.** AMD Radeon RX 6700 XT, natively gfx1031, run as gfx1030 through
`HSA_OVERRIDE_GFX_VERSION=10.3.0` (already in `~/.bashrc`). ROCm 6.4.1 at
`/opt/rocm-6.4.1` (no `/opt/rocm` symlink). Venv at `./.venv` (15 GB): torch
2.9.1+rocm6.4, pytorch-triton-rocm 3.5.1. Torch's flash and mem-efficient SDPA
are absent for this arch. Small models and the venv live on the root NVMe;
`HF_HOME` and bulk trace output go to `/mnt/data/reproducible-ai/` (second
NVMe, NTFS via FUSE, 1.1 TB free). SGLang, vLLM and tinygrad cannot run on this
GPU (`docs/research/local-gpu-feasibility.md`).

**Cloud.** Lambda Cloud only, provisioned by the user when a phase is ready
(`docs/research/cloud-gpu-providers.md`). H100 SXM is the reference SKU, H100
PCIe a supported second envelope (test 7), B200 the NVFP4 reference, 2x/4x H100
SXM for the TP phase. One pinned container image across all of them. Envelope
capture script and PCI device ids are in the provider doc. Budget for the
validation campaign is roughly $180-250.

## 14. Phase plan

- **P0** Repo skeleton, trace library (schema, hashing, envelope capture,
  KDF/RNG reference), reference oracle v0 in plain torch with contiguous KV and
  the seeded sampler, `record` and `verify` CLIs, HF tolerance oracle, tests 1
  and 4. Qwen3-0.6B, local.
- **Spike** (first cloud task, one H100, a few hours): characterize SGLang's
  deterministic mode against tests 2, 3, 5, 6 and sampler tie handling. Its
  result decides how much of P1 is needed.
- **P1** Paged KV, continuous-batching scheduler, chaos batcher, tests 2, 3
  and 6 in the oracle; Thinking Machines kernels wired in where a
  batch-invariant op is needed locally; stable-tie-break sampler kernel.
- **P2** Counter-based RNG in Triton, SR epilogue GEMM, emulated FP8 and NVFP4
  quantizers, SR mask reference tests, `output_cast: stochastic` end to end.
- **P3** Cloud H100: oracle on NVIDIA, tests 5 and 7, real FP8 W8A8 and FP8 KV,
  SGLang adapter (trace writer + verifier backend), overhead measurement vs
  stock SGLang. vLLM adapter if time allows.
- **P4** B200 NVFP4, then MoE with Qwen3-30B-A3B.
- **P5** Tensor parallel with pinned collectives; `parallel_config` becomes
  non-trivial.

Upstreaming (the trace format first, kernels second) is out of scope until the
oracle passes tests 1-7.

## 15. Non-goals

Cross-architecture or cross-vendor bitwise identity. Throughput optimization
before determinism is proven (overhead is measured and reported, target within
the tens-of-percent range of prior work). HTTP serving in the oracle.
Speculative decoding. Prefix caching in v1. Training (the RNG contract and
schema are designed so a training step could reuse them; nothing is
implemented). Cryptographic proof, attestation or TEEs. Retention and
zero-data-retention policy (noted as a consumer-side concern in the research
doc, not designed here). Weight-only int4.

## 16. Risks and open questions

- Whether torch elementwise ops stay invariant across torch versions within an
  envelope is empirical; the soft-first policy is designed to learn this.
- Triton on RDNA2 is not in Triton's CI; a kernel may need a HIP fallback.
- SGLang's radix-cache prefill parity is an open upstream issue; the adapter
  may have to disable the radix cache or accept a documented gap.
- NVFP4 on B200 depends on kernels (cuBLASLt / CUTLASS / FlashInfer) whose
  reduction order we do not control; the quantizer is ours, the GEMM may not
  be. This is P4's central question.
- SGLang adapter details found in the audit: the default seed is 42 and `n > 1`
  does not bump it (the adapter must set an explicit seed per sample); `min_p`
  with a seed is a hard assert; the deterministic attention backend is chosen
  by GPU architecture (FA3 on SM90, FlashInfer on SM100/120), so the adapter
  must record the attention backend as part of the kernel config; full-vocab
  evidence comes back as temperature-scaled log-softmax (via
  `top_logprobs_num = vocab_size`) rather than raw logits, so step records
  carry a `logits_hash_kind` field and the verifier compares like with like.

## Appendix: decision log

| Q | Decision |
|---|---|
| 1 | Determinism claim (b): bitwise logits, fixed envelope |
| 2 | (c) reference oracle locally + engine integration on cloud; SGLang cannot run locally |
| 3 | Qwen3-0.6B and SmolLM2; MoE later |
| 4 | Replay from step 0 and resume from step k |
| 5 | Seeded sampling from day one |
| 6 | bf16 locally; FP8 (H100) and NVFP4 (B200) on cloud |
| 7 | Single GPU first, TP later |
| 8 | Lambda Cloud, H100 SXM + PCIe, B200 |
| 9 | Triton primary (verified on gfx1030), HIP fallback only if needed; tinygrad rejected |
| 10 | Per step: token id + BLAKE3-128 of fp32 logits |
| 11 | Refuse on hard-envelope mismatch; `--no-verify` resume path |
| 12 | Research artifact; training reuse designed, not built |
| 13 | MoE cloud-only with Qwen3-30B-A3B |
| 14 | FP8 W8A8 + FP8 KV on H100; NVFP4 W4A4 on B200; no weight-only int4 |
| 15 | Session-shaped trace with per-turn hash chain |
| 16 | Both fast resume and exact replay |
| 17 | Root seed with derived, recorded, overridable subkeys |
| 18 | Hash fp32 pre-temperature logits |
| 19 | Tests 1-6, later 7 |
| 20 | Paged KV; contiguous allowed in v0 |
| 21 | Both file and in-memory weight hashes |
| 22 | MIT; package `reproducible_ai` |
| 23 | Two entry points, `record` and `verify` |
| 24 | OS-random root seed by default, external accepted, provenance recorded |
| 25 | Manifest object in header; image digest from cloud phase |
| 26 | Soft where possible |
| 27 | Batch-invariant RNG element identity; Philox4x32-10 |
| 28 | Reductions custom, elementwise in torch under contract tests |
| 29 | Own kernels where needed, TML kernels as reference |
| 30 | Precision contract as in section 9; fp16 allowed |
| 31 | Offline API + CLI, no HTTP |
| 32 | JSONL with safetensors sidecar |
| 33 | Packet extraction: schema guarantee only |
| 34 | Phase order as in section 14 |
| 35 | Lambda only |
| 36 | H100 SXM reference, PCIe supported |
| 37 | SGLang adapter first, vLLM later |
| 38 | Upstreaming out of scope for research phase |
| 39 | (c) minimal oracle, no own bf16 kernels, spike first |
| 40 | Six hard envelope fields, compiled-kernel registry hash |
| 41 | Defaults for 25, 28, 33 confirmed |
