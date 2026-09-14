# SGLang deterministic-inference audit

Audited against `sgl-project/sglang` `main` at commit
[`0027af2eace5ccc2116c8c993ce71aebf1535264`](https://github.com/sgl-project/sglang/commit/0027af2eace5ccc2116c8c993ce71aebf1535264)
(2026-09-09). All paths and line numbers below refer to that commit; links are
pinned to it. Blob links: `https://github.com/sgl-project/sglang/blob/0027af2e/<path>#L<n>`.

Question index: [1 Seed to noise](#1-how-the-per-request-seed-becomes-sampling-noise) ·
[2 Step index](#2-how-the-step-index-t-is-tracked) ·
[3 Top-k/p/min-p](#3-top-k--top-p--min-p-in-deterministic-mode) ·
[4 Run envelope](#4-run-envelope-what-a-third-party-would-need-to-replay) ·
[5 Logits exposure](#5-per-step-logits--logprobs-exposure) ·
[6 Documented limitations](#6-documented-limitations) ·
[7 Cross-machine claims](#7-cross-machine--cross-sku-reproducibility)

---

## 1. How the per-request seed becomes sampling noise

**Short answer:** In deterministic mode the Gumbel noise for request *r* at a
given step is a pure function of `(r.sampling_seed, absolute_position, vocab_index)`.
It is computed by a counter-based hash (MurmurHash3-32) per `(row, column)`
element; there is no `torch.Generator`, no Philox offset, and no
`torch.rand`/`torch.multinomial` over the `[batch, vocab]` tensor. Batch
position and the identity of co-batched requests do not enter the hash.

### 1a. Seed plumbing

`sampling_seed` is a request-level sampling param
([`python/sglang/srt/sampling/sampling_params.py:153`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/sampling/sampling_params.py#L153)):

```python
    sampling_seed: Optional[int] = None
```

The OpenAI-compatible `seed` field maps straight onto it
([`entrypoints/openai/protocol.py:1143`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/entrypoints/openai/protocol.py#L1143),
[`entrypoints/openai/serving_completions.py:171`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/entrypoints/openai/serving_completions.py#L171)):

```python
            "sampling_seed": self.seed,
```

Per batch, `SamplingBatchInfo.from_schedule_batch` builds a `[batch]` int64
tensor of seeds, defaulting to **42** when the request did not set one, and
only when deterministic mode is on
([`sampling/sampling_batch_info.py:117-132`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/sampling/sampling_batch_info.py#L117-L132)):

```python
        sampling_seed = (
            torch.tensor(
                [
                    (
                        r.sampling_params.sampling_seed
                        if r.sampling_params.sampling_seed is not None
                        else 42
                    )
                    for r in reqs
                ],
                dtype=torch.int64,
                pin_memory=_pin,
            ).to(device, non_blocking=True)
            if enable_deterministic
            else None
        )
```

The tensor is row-indexed alongside the batch under `filter_batch` (retraction,
finish) and `merge_batch` (new prefill joins running batch)
([`sampling_batch_info.py:324-333`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/sampling/sampling_batch_info.py#L324-L333),
[`:446-456`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/sampling/sampling_batch_info.py#L446-L456)):

```python
        for item in ["temperatures", "top_ps", "top_ks", "min_ps", "sampling_seed"]:
            value = getattr(self, item, None)
            if value is not None:
                setattr(self, item, value[keep_indices_device])
```

Consequence worth flagging: with `n > 1` the same `sampling_params` dict is
replicated to every parallel sample
([`managers/io_struct.py:689-696`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/managers/io_struct.py#L689-L696)),
and nothing bumps the seed per sample (grep for `sampling_seed` under
`managers/` and `entrypoints/` finds only the two OpenAI mappings above). So in
deterministic mode, `n>1` with a fixed (or default 42) seed yields *n*
identical completions; the docs' GRPO recipe indeed sends separate requests with
distinct seeds.

### 1b. The noise construction

All seeded sampling funnels into `multinomial_with_seed`
([`layers/sampler.py:850-891`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/sampler.py#L850-L891)):

```python
def multinomial_with_seed(
    logprobs: torch.Tensor, seed: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    ...
    n, m = logprobs.shape
    seed = seed.to(torch.uint64)
    col_indices = torch.arange(m, device=logprobs.device)
    hashed = murmur_hash32(seed, positions, col_indices)

    # NOTE (sehoon): it is critical to keep gumbel noise calculation in float64 to avoid numerical instability.
    # keeping logprobs in float64 is less critical, but we found it's still safer to keep it in float64.
    x = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max

    # x is a uniform sample in [0, 1]. get gumbel noise from it.
    # which is equivalent to -log(-log(x))
    # keep everything in in-place operations to avoid unnecessary memory allocations.
    # clamp both ends: x == 1 gives gumbel +inf (NaN at -inf logprobs); the cap is
    # the hash spacing so that bucket matches its neighbor instead of dominating
    x.log_().clamp_(min=torch.finfo(x.dtype).min, max=-(2.0**-32)).neg_()
    x.log_().neg_()  # -log(-log(x)) == gumbel noise

    # add gumbel noise to logprobs
    x.add_(logprobs.to(torch.float64))

    return torch.argmax(x, dim=1, keepdim=True)
```

The hash kernel is a Triton MurmurHash3-32 over four 32-bit blocks
`(seed_lo, seed_hi, position, column)` — one program per row, one uint32 per
`(row, column)`, no cross-row state
([`python/sglang/kernels/ops/sampling/murmur_hash.py:50-101`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/kernels/ops/sampling/murmur_hash.py#L50-L101)):

```python
    """
    We treat 64-bit seed, 32-bit position, and 32-bit col_index as 4 4-byte blocks, and bit-blend them together.
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    row_idx = pid_row
    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ...
    seed = tl.load(seed_ptr + row_idx).to(tl.uint64)
    pos = tl.load(positions_ptr + row_idx).to(tl.uint32)
    col = tl.load(col_indices_ptr + col_offsets, mask=mask, other=0).to(tl.uint32)

    h: tl.uint32 = 0  # hash accumulator
    k: tl.uint32 = (seed & 0xFFFFFFFF).to(tl.uint32)
    h = murmur3_mix(h, k)
    k = ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
    h = murmur3_mix(h, k)
    h = murmur3_mix(h, pos)
    h = murmur3_mix(h, col)
    # Finalize (len=16 for seed + pos + col)
    h ^= 16
    h = fmix32(h)
    tl.store(output_ptr + row_idx * num_cols + col_offsets, h, mask=mask)
```

A CPU path calls `torch.ops.sgl_kernel.murmur_hash32_cpu` with the same
signature ([`murmur_hash.py:112-117`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/kernels/ops/sampling/murmur_hash.py#L112-L117)).

So for a fixed `(seed, position)` the noise vector over the vocab is a
deterministic integer function; the only floating-point work is
`uint32 -> float64 / UINT32_MAX`, two `log`s in float64, an add, and an
`argmax`. The `argmax` is over the row only, so ties/ordering depend on nothing
outside the row. `torch.argmax` returns the first maximal index, and the
perturbed values are float64 so exact ties are practically impossible.

Note `sampling_seed` is `int64` but only its low 64 bits as `uint64` feed the
hash; negative seeds are allowed and wrap.

### 1c. Which code paths use it

`Sampler.forward` receives `positions` and threads it into all sampling
branches ([`layers/sampler.py:123-145`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/sampler.py#L123-L145)):

```python
            positions: The positions of the tokens in the sequence. Used for deterministic sampling
                to get the unique seed for each position.
```

- Greedy (`is_all_greedy`): `torch.argmax(logits, -1)` — no noise
  ([`:163-171`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/sampler.py#L163-L171)).
- Simple case (no top-k/p/min-p): `sampling_from_probs_torch` ->
  `multinomial_with_seed(torch.log(probs), sampling_seed, positions)`
  ([`:894-920`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/sampler.py#L894-L920)).
  With `--rl-on-policy-target` set, `_sample_from_logprobs` is used on a
  `log_softmax` of bf16 temperature-scaled logits instead
  ([`:190-223`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/sampler.py#L190-L223),
  [`:566-583`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/sampler.py#L566-L583)).
- Top-k/top-p/min-p: the `pytorch` sampling backend is forced (see §3); the
  FlashInfer backend asserts the seed is absent
  ([`:358-361`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/sampler.py#L358-L361)):

  ```python
                assert sampling_info.sampling_seed is None, (
                    "Sampling seed is not supported for flashinfer backend"
                )
  ```

- Non-deterministic mode (seed `None`) uses `torch.multinomial`, or an
  unseeded `exponential_()` Gumbel-max when `SGLANG_OPT_USE_GUMBEL_SAMPLE` is
  set ([`:904-913`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/sampler.py#L904-L913)).
  That path is the only global-RNG draw and is not batch-invariant.

The EAGLE verify path mirrors the same construction for rejection coins
(`murmur_hash32(seed, seq_lens, column)`), but its docstring states the draft
side is still unseeded
([`speculative/eagle_utils.py:618-637`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/speculative/eagle_utils.py#L618-L637)):

```python
    Scope: this seeds only the verify-side RNG. With rejection sampling the
    draft workers still pick candidates via unseeded multinomial
    (fast_sample in eagle_worker_v2), so that mode stays non-deterministic
    until the draft RNG is seeded in a follow-up; top-k/greedy draft
    selection is already deterministic.
```

---

## 2. How the step index `t` is tracked

**Short answer:** there is no per-request counter. The "step" fed to the hash is
the **absolute token position** of the token whose logits are being sampled,
i.e. `seq_len - 1` = `len(prompt) + len(output_so_far) - 1`. Because it is
derived from sequence length rather than from scheduler history, retraction,
chunked prefill and radix-cache hits all reproduce the same position and hence
the same noise. Whether the *logits* match is a separate (batch-invariance)
question.

The call site in the model runner
([`model_executor/model_runner.py:1915-1928`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/model_executor/model_runner.py#L1915-L1928)):

```python
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
            # For prefill, we only use the position of the last token.
            (
                forward_batch.positions
                if forward_batch.forward_mode.is_decode()
                else forward_batch.seq_lens - 1
            ),
        )
```

- **Decode:** `forward_batch.positions = clamp_position(batch.seq_lens)`
  ([`model_executor/forward_batch_info.py:925`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/model_executor/forward_batch_info.py#L925)),
  where `clamp_position` is `clamp(seq_lens - 1, min=0)`
  ([`kernels/ops/attention/clamp_position.py:27-30`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/kernels/ops/attention/clamp_position.py#L27-L30)).
- **Extend/prefill:** `seq_lens - 1`, with `seq_lens = [r.extend_range.end ...]`
  ([`managers/schedule_batch.py:2572`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/managers/schedule_batch.py#L2572))
  where `extend_range.end` indexes into `full_untruncated_fill_ids =
  origin_input_ids + output_ids`
  ([`:1438`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/managers/schedule_batch.py#L1438),
  [`:1414-1418`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/managers/schedule_batch.py#L1414-L1418)).

Cases:

- **Retraction / preemption:** `Req.reset_for_retract` clears prefix indices,
  KV bookkeeping and `extend_range`, but does not touch `output_ids` or
  `sampling_params`
  ([`schedule_batch.py:1815-1852`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/managers/schedule_batch.py#L1815-L1852)).
  On re-schedule the request is re-prefilled over `prompt + output_ids`, so the
  next sampled position is again `len(prompt)+len(output_ids)-1` and the noise
  is identical. The response even carries `num_retractions` in `meta_info`
  ([`managers/tokenizer_manager.py:2272`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/managers/tokenizer_manager.py#L2272)).
- **Chunked prefill:** only the final chunk has `extend_range.end ==
  len(fill_ids)`; for earlier chunks `seq_lens - 1` points mid-prompt, but the
  result processor only appends the sampled token when
  `req.inflight_middle_chunks <= 0`; while a request is "being chunked" its
  prefill "is not finished" and the sample is dropped
  ([`managers/scheduler_components/batch_result_processor.py:328-392`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L328-L392)),
  so mid-chunk hash positions are never consumed. Deterministic mode also aligns
  chunk boundaries to a fixed tile size so the *attention* reduction order is
  chunk-invariant (`SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE=4096`,
  `SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE=4096`;
  [`managers/scheduler.py:1671-1687`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/managers/scheduler.py#L1671-L1687)).
- **Radix-cache hit:** position is derived from full sequence length, not from
  `extend_prefix_lens`, so a cached vs. uncached prefill of the same tokens
  hashes to the same position. The `radix_cache` test mode exists precisely to
  check that logits also agree
  ([`python/sglang/test/test_deterministic.py:560-641`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/test/test_deterministic.py#L560-L641)).

Implication for a verifier: the noise sequence is **content-independent**. Two
different prompts of equal length with the same seed receive identical Gumbel
vectors at each position; only the logits differ. Also, `seed` defaults to 42
for every request that omits it, so "unseeded" traffic is fully replayable.

---

## 3. Top-k / top-p / min-p in deterministic mode

**Short answer:** deterministic mode forces the `pytorch` sampling backend,
which does `probs.sort(descending=True)` (unstable, GPU) followed by cumulative
masking and then the seeded Gumbel-max over the *sorted, masked* row. The
Gumbel noise is indexed by the **sorted column**, so a sort-order permutation
among exactly-equal probabilities would change which token receives which
noise value. In practice ties in float32 softmax outputs are rare but not
impossible (e.g. many `0.0` entries after masking are harmless because their
log is `-inf`; ties among *surviving* non-zero probabilities are the concern).
`min_p` with a seed is explicitly asserted unsupported.

Backend forcing
([`arg_groups/overrides.py:1111-1118`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/arg_groups/overrides.py#L1111-L1118)):

```python
@register_post_process
def _deterministic_sampling_backend(view: Any) -> dict:
    if view.enable_deterministic_inference and view.sampling_backend != "ascend":
        logger.warning(
            "Sampling backend is set to pytorch for deterministic inference."
        )
        return {"sampling_backend": "pytorch"}
    return {}
```

The torch implementation
([`layers/sampler.py:717-774`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/sampler.py#L717-L774)):

```python
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        # TODO: probs_sort should be re-normalized for the use of multinomial_with_seed
        assert sampling_seed is None, (
            "With sampling seed, multinomial_with_seed will provide wrong results"
        )
        ...
    if sampling_seed is None:
        sampled_index = torch.multinomial(probs_sort, num_samples=1)
    else:
        # NOTE: when using top-k/top-p/min-p sampling, we need to modify probs before we
        # apply log to get logprobs. Therefore, we cannot use log_softmax directly.
        # For now, we use log to the modified probs to get logprobs, but for numerical
        # stability, we'd better come up with a solution to use log_softmax.
        logprobs = probs_sort.to(torch.float64, copy=return_filtered_probs)
        ...
        logprobs.log_()
        sampled_index = multinomial_with_seed(logprobs, sampling_seed, positions)
    ...
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
```

Observations:

- `torch.sort` is called without `stable=True`. Row-wise sort on CUDA is
  segmented and does not depend on other rows, so it is batch-invariant in the
  sense that other rows cannot affect it, but among equal keys the output
  permutation is implementation-defined.
- The Gumbel column index is the *rank*, not the token id (`col_indices` is
  `arange(m)` over the sorted tensor), so the noise/token pairing is only as
  stable as the sort's tie handling. A Gumbel-max over the unsorted vocab with
  a top-k/top-p mask would have avoided this; SGLang does not do that.
- The top-k cutoff is applied after `cumsum` on the sorted vector; the
  `cumsum` itself is a float32 scan whose result can depend on kernel
  choice/arch, which decides the top-p boundary token. This is a numeric
  (not RNG) portability concern.
- The simple (no top-k/p) path has no sort at all and is fully
  batch-invariant given batch-invariant logits.
- `min_p` + seed: hard `assert` (server would crash) — effectively
  unsupported.
- Ascend (NPU) has its own variant with the same structure
  ([`:777-848`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/sampler.py#L777-L848)).

Output-logprob top-k for `top_logprobs_num` uses `logprobs.topk(max_k)`
([`layers/logprob_processor.py:93-94`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/logprob_processor.py#L93-L94)),
again without an explicit tie-break.

For DeepSeek-V3.2 sparse attention the *indexer* top-k has separate knobs:
`SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC` and
`SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK={small,large}`
([`docs/docs/references/environment_variables.mdx:530-538`](https://github.com/sgl-project/sglang/blob/0027af2e/docs/docs/references/environment_variables.mdx#L530-L538)).

---

## 4. Run envelope: what a third party would need to replay

**Short answer:** SGLang records launch configuration and its own package
version, plus an operator-supplied `weight_version` string. It does **not**
record a model weight hash, git SHA, kernel-library versions (flashinfer,
sgl-kernel, triton, torch, CUDA), driver, or GPU model/arch in any request,
response, or endpoint.

`/server_info` (and deprecated `/get_server_info`)
([`entrypoints/http_server.py:811-842`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/entrypoints/http_server.py#L811-L842)):

```python
    return msgspec_to_builtins(
        {
            **server_args.resolved_dict(),
            "launch_command": server_args.launch_command,
            **_global_state.scheduler_info,
            "startup_time": _global_state.tokenizer_manager.startup_time,
            "internal_states": internal_states,
            "version": __version__,
            "kv_events": describe_kv_events_publisher(server_args),
        }
    )
```

- `resolved_dict()` is the fully-resolved `ServerArgs` (so it does capture the
  post-resolution `attention_backend`, `sampling_backend=pytorch`,
  `disable_radix_cache`, `enable_deterministic_inference`, `quantization`,
  `dtype`, `tp_size`, etc.) — useful, but only *configuration*
  ([`server_args.py:290-301`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/server_args.py#L290-L301)).
- `scheduler_info` is `max_total_num_tokens`, `max_req_input_len`,
  `startup_time` ([`managers/scheduler.py:1819-1822`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/managers/scheduler.py#L1819-L1822)).
- `version` is the pip/setuptools-scm version string only
  ([`python/sglang/version.py`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/version.py)).

`/model_info` reports `model_path`, `served_model_name`, `tokenizer_path`,
`weight_version` (operator string, changed by weight updates), `load_format`,
`model_type`, `architectures`, parsers
([`http_server.py:753-778`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/entrypoints/http_server.py#L753-L778)).

Per-response `meta_info` contains `id`, `finish_reason`, `prompt_tokens`,
`completion_tokens`, `weight_version`, `num_retractions`, timing, and (when
requested) logprob arrays
([`tokenizer_manager.py:2268-2272`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/managers/tokenizer_manager.py#L2268-L2272)).
No sampling seed is echoed back, so a replayer must know that an omitted seed
means 42.

A grep for `git_sha|git_commit|model_hash|weights_hash` under `python/sglang/srt`
finds nothing relevant; the only `get_device_name()` use is in a loads
endpoint ([`entrypoints/v1_loads.py:41`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/entrypoints/v1_loads.py#L41)).

Deterministic mode also mutates process environment that is *not* reflected in
`/server_info`: `NCCL_ALGO=allreduce:tree`, `NCCL_MIN/MAX_NCHANNELS`
(`SGLANG_DETERMINISTIC_NCCL_NCHANNELS`, default 8),
`SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0`
([`arg_groups/attention_hook.py:599-632`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/arg_groups/attention_hook.py#L599-L632),
[`arg_groups/serving_hook.py:401-405`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/arg_groups/serving_hook.py#L401-L405),
[`environ.py:1243`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/environ.py#L1243)).

---

## 5. Per-step logits / logprobs exposure

**Short answer:** SGLang can return the sampled token's logprob, the top-`k`
logprobs (`top_logprobs_num`, no enforced upper bound — `k` may equal the vocab
size), and logprobs for an arbitrary caller-supplied list of token ids
(`token_ids_logprob`), for both prompt and generated positions. There is no
"return full logits" request option; `return_full_logits` exists only as a
`LogitsProcessor` constructor flag used by diffusion-LLM models.

Request fields ([`managers/io_struct.py:226-233`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/managers/io_struct.py#L226-L233)):

```python
    return_logprob: Optional[Union[List[bool], bool]] = None
    logprob_start_len: Optional[Union[List[int], int]] = None
    top_logprobs_num: Optional[Union[List[int], int]] = None
    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None
```

plus `return_flat_raw_top_logprobs` / `_b64` for compact prompt top-k arrays
([`:238-242`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/managers/io_struct.py#L238-L242)).
Top-k is computed with `logprobs.topk(max_k, dim=-1)` over the whole vocab
([`layers/logprob_processor.py:93-94`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/logprob_processor.py#L93-L94));
no validation caps `top_logprobs_num` (grep for `top_logprobs_num` with
`raise`/`max` under `managers/` and `entrypoints/` finds only `> 0` checks).
So a verifier can request `top_logprobs_num = vocab_size` (or pass
`token_ids_logprob = list(range(vocab_size))`) to obtain the full log-softmax
row per step, at a large bandwidth cost. Values are Python floats (float32
precision through `.tolist()`), not raw bf16 logits.

`return_full_logits` is internal
([`layers/logits_processor.py:381,414,1063-1074`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/logits_processor.py#L1063-L1074))
and set only by `sdar.py`, `sdar_moe.py`, `llada2.py`.

Deterministic-mode logprob semantics that matter for hashing:

- Returned logprobs are **temperature-scaled** (unless
  `SGLANG_RETURN_ORIGINAL_LOGPROB` is set) and are computed via
  `F.log_softmax` specifically so prefill and decode logprobs agree bitwise
  ([`layers/sampler.py:228-241`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/sampler.py#L228-L241)):

  ```python
                # Deterministic inference must derive the returned logprobs
                # from F.log_softmax — the same kernel prefill rescoring uses —
                # not log(softmax(x)) below: the two disagree at ~1e-6 despite
                # being mathematically equivalent, which breaks bitwise
                # prefill/decode logprob alignment.
  ```

- The "fast input logprobs" path is disabled in deterministic mode for the
  same reason ([`layers/logprob_processor.py:449-458`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/logprob_processor.py#L449-L458)).
- `--rl-on-policy-target` switches to bf16 `logits / T` then `log_softmax`
  ([`sampler.py:190-200`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/sampler.py#L190-L200)).

`return_hidden_states` and `return_routed_experts` are also available (server
flags gate them) and could serve as additional fingerprints
([`io_struct.py:248-252`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/managers/io_struct.py#L248-L252)).

---

## 6. Documented limitations

### 6a. Docs page `docs/docs/advanced_features/deterministic_inference.mdx`

[Source](https://github.com/sgl-project/sglang/blob/0027af2e/docs/docs/advanced_features/deterministic_inference.mdx)
(rendered at https://docs.sglang.ai/advanced_features/deterministic_inference). Verbatim:

> Building on Thinking Machines Lab's batch-invariant operators, SGLang achieves fully deterministic inference while maintaining compatibility with chunked prefill, CUDA graphs, radix cache, and non-greedy sampling.

> Deterministic inference is only supported with the following three attention backends: **FlashInfer**, **FlashAttention 3 (FA3)**, and **Triton**.

Compatibility table (lines 30-70):

| Attention Backend | CUDA Graph | Chunked Prefill | Radix Cache | Non-greedy Sampling (Temp > 0) |
|---|---|---|---|---|
| FlashInfer | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| FlashAttention 3 (FA3) | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| Triton | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |

> By default, SGLang uses a sampling seed of `42` for reproducible sampling

> This approach ensures that:
> - Different seeds produce diverse responses
> - The same seed always produces the same response across different runs
> - Results are reproducible for debugging and evaluation

> Expected result: All tests should show `Unique samples: 1` (perfectly deterministic).

The page contains **no** performance numbers, no quantization statement, no
MoE/EP/speculative-decoding statement. The `--attention-backend` row says
"string; default: fa3".

### 6b. Other docs pages

- `server_arguments.mdx`: `--enable-deterministic-inference` — "Enable
  deterministic inference mode with batch invariant ops."; `--bf16-gemm-backend`
  — "`auto` (default; selects `cutedsl` on SM10x GPUs, except deterministic
  inference selects `torch`; …)"; `--triton-attention-split-tile-size` — "The
  size of split KV tile in flash decoding Triton kernel. Used for deterministic
  inference."; `--mlx-enable-sampling` — "per-request sampling_seed under
  --enable-deterministic-inference (deterministic within MLX only)"
  ([lines 58, 719, 1491, 2606, 2719](https://github.com/sgl-project/sglang/blob/0027af2e/docs/docs/advanced_features/server_arguments.mdx)).
- `environment_variables.mdx` "Deterministic Inference" table:
  `SGLANG_ENABLE_DETERMINISTIC_INFERENCE` ("fixed reduction/accumulation
  order"), `SGLANG_USE_1STAGE_ALLREDUCE` (AMD; "auto-enabled when deterministic
  inference is on"), `SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE=4096`,
  `SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE=2048`,
  `SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE=4096`,
  `SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE=256`; and
  `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE` — "Deterministic inference overrides
  this to `false`."
  ([lines 1647-1694, 773-775](https://github.com/sgl-project/sglang/blob/0027af2e/docs/docs/references/environment_variables.mdx#L1647-L1694)).
- `speculative_decoding.mdx` (UNO): "Grammar decoding, returned logprobs or
  hidden states, sampling penalties, `min_p`, logit bias, custom logit
  processors, strict thinking, and deterministic inference are not yet
  supported." ([line 521](https://github.com/sgl-project/sglang/blob/0027af2e/docs/docs/advanced_features/speculative_decoding.mdx#L521)).
- `piecewise_cuda_graph.mdx`: piecewise (prefill) CUDA graph is auto-disabled
  for "Deterministic inference" ([line 236](https://github.com/sgl-project/sglang/blob/0027af2e/docs/docs/advanced_features/piecewise_cuda_graph.mdx#L236)).
  Decode CUDA graphs remain enabled.
- `faq.mdx`: "To achieve more deterministic outputs in the current code, you
  can add `--disable-radix-cache` and send only one request at a time. The
  results will be mostly deterministic under this setting. … Recently, we also
  introduced a deterministic mode, you can enable it with
  `--enable-deterministic-inference`."
- `sglang_for_rl.mdx`: "SGLang supports a deterministic inference mode that
  reduces non-determinism across batch shapes. This mitigates variance
  introduced by runtime batching and kernel selection. To further achieve true
  on-policy training, you need to modify the training engine to use the same
  deterministic kernels." ([line 648](https://github.com/sgl-project/sglang/blob/0027af2e/docs/docs/advanced_features/sglang_for_rl.mdx#L648)).
- Ascend `support_features.mdx`: rejection sampling "cannot be combined with
  non-default speculative acceptance thresholds or deterministic inference."

### 6c. Tracking issue [#10278](https://github.com/sgl-project/sglang/issues/10278)

"[Feature] Support deterministic inference with Batch Invariant Ops" — opened
2025-09-10, **auto-closed for inactivity 2026-02-21**, 22 comments. Checklist
verbatim (unchecked = open when it closed):

```
### Backbone: Attention Backend, batch_invariant library integration
- [x] Support cuda graph #10645
- [x] Support temperature > 0 #10678
- [x] Attention Backend: Flashinfer: flashinfer#1675 #10645
- [x] Attention Backend: Triton #10425 #10694
- [x] Attention Backend:  FA3 #10651
- [x] Move batch invariant triton kernels to sglang repo #10695
- [x] Add unit tests #11095  #10994 #11368

### Communication (NCCL)
- [x] Support tensor parallelism with deterministic all-reduce kernels #10785 #10930
- [x] Support tensor parallelism with deterministic all-reduce kernels (AMD) #15340
- [ ] Not deterministic on Blackwell for TP4 #11513

### Radix Cache Support
- [x] FA3 supported given its already 1-stage prefill at the beginning
- [x] Triton Supported by PR: #11147
- [ ] FlashInfer Support
- [ ] Making Prefill with Radix Cache has the same output as Prefill without Radix cache

### Model Support
- [x] Qwen3 Dense
- [x] Qwen3 MOE
- [x] Deepseek v3 Deterministic Support #12095
- [ ] Qwen3 Next (Linear Attention) #12845

### Quantization
- [ ] Blockwise FP8 Kernel #11491
- [ ] per-channel fp8 Gemm
- [ ] fp8 fused MoE
- [ ] nvfp4 Gemm/MoE

### Parallelism
- [ ] Support DP Attention #11023
- [ ] Support EP

### Spec Decoding
- [ ] Speculative decoding drafters #8391 #9877

### Perf
- [x] Accelerate batch invariant triton kernels #12142 #12144
```

Relevant comments:

- Fridge003, 2025-09-23: "Do you mean supporting deterministic inference for
  spec decoding? We don't have that plan by now."
- Fridge003, 2025-10-14: "SGLang will avoid using autotune. For triton kernels,
  we will fix configurations like block sizes or num warps for the same model,
  so deterministic results can be obtained."
- Fridge003, 2025-10-24 (FlashInfer + radix, PR #10721): "We detected some
  indeterminism after merging this PR, so it was temporarily reverted #11728."
- hlu1, 2025-10-25: "The current custom topk kernel, both unfused and fused, is
  non-deterministic which results in variance in accuracy benchmarks."
  (DeepSeek-V3.2 indexer top-k.)
- jzft-nuosu, 2025-12-22: DeepSeek-V3.1 on 8xH200 with
  `--enable-deterministic-inference --attention-backend fa3 --speculative-algorithm EAGLE`:
  "the decoding speed has dropped from 100 tokens/s to 60 tokens/s."

### 6d. LMSYS blog (2025-09-22) — the only published overhead numbers

https://lmsys.org/blog/2025-09-22-sglang-deterministic/ (linked from `faq.mdx`):

- End-to-end latency, deterministic vs normal, per backend (1K/1K, 4K/4K,
  8K/8K): FlashInfer "+42.6%", "+46.0%", "+24.4%"; FA3 "+27.2%", "+30.2%",
  "+35.7%"; Triton "+55.1%", "+44.64%", "+44.80%". Average "34.35%" slowdown
  (FlashInfer + FA3).
- CUDA graph on vs off (deterministic): FlashInfer "1245.51 (2.82x)" vs 441.73;
  FA3 "1247.64 (2.79x)" vs 447.64; Triton "1228.36 (2.93x)" vs 419.64.
- "Radix cache is disabled for Flashinfer and Triton since their support is
  still in progress" / "We disabled radix cache for all performance benchmarks".
- At publication: "only support deterministic inference for dense models";
  TP1/TP2 verified, "larger TP setups require modifications".
- Sampling: "this operator perturbs logits with Gumbel noise generated from a
  **seeded hash function**. As a result, the same `(inputs, seed)` pair always
  yields the same sample, even when temperature > 0."

### 6e. Limitations enforced in code (not all documented)

Resolution pipeline
([`arg_groups/attention_hook.py:512-632`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/arg_groups/attention_hook.py#L512-L632),
[`arg_groups/overrides.py:1111-1188`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/arg_groups/overrides.py#L1111-L1188),
[`arg_groups/choices.py:118-125,250`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/arg_groups/choices.py#L118-L125)):

- Allowed attention backends: `DETERMINISTIC_ATTENTION_BACKEND_CHOICES =
  ["ascend", "fa3", "fa4", "flashinfer", "intel_xpu", "triton"]`; default is
  `fa3` on SM90 and older, `flashinfer` on SM100/SM120 (or `triton` for
  DeepSeek archs there).
- Radix cache is force-disabled unless backend is in
  `RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND = ["ascend", "fa3", "fa4", "triton"]`:
  "Currently radix cache is not compatible with {attention_backend} attention
  backend for deterministic inference. It will be supported in the future."
- DeepSeek/MLA models must use a radix-supported backend; `fa4` requires
  SM100/SM110.
- TP>1 on CUDA: `NCCL_ALGO=allreduce:tree`, channel count pinned,
  `disable_custom_all_reduce=True`, `enable_torch_symm_mem=False`,
  `flashinfer_allreduce_fusion_backend=None`; AMD uses 1-stage all-reduce.
- Sampling backend forced to `pytorch`.
- `--rl-on-policy-target` implies deterministic mode.
- Speculative decoding: UNO rejected ("its sampling path does not use
  per-request seeds"); `--speculative-use-rejection-sampling` rejected ("the
  sampling kernel draws coins from the global RNG and is not batch-invariant")
  ([`arg_groups/speculative_hook.py:342-346,937-942`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/arg_groups/speculative_hook.py#L937-L942)).
  EAGLE itself is allowed and seeds only the verify coins (see §1c).
- Kernels: `--bf16-gemm-backend cutedsl` rejected ("batch-size dependent")
  ([`layers/quantization/unquant.py:236-241`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/quantization/unquant.py#L236-L241));
  FlashInfer autotune skipped ("Tuned configs are per problem shape, so the
  reduction order would follow the batch shape")
  ([`model_executor/runner/flashinfer_autotune.py:64-67`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/model_executor/runner/flashinfer_autotune.py#L64-L67));
  fused-MoE Triton config pinned to `BLOCK_SIZE_M=64, N=64, K=32, GROUP_SIZE_M=8`
  and tuned JSON configs ignored ("Deterministic inference is enabled, using
  default MoE kernel config.")
  ([`fused_moe_triton_config.py:72-76,190-197`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py#L190-L197));
  tensor-core MoE router disabled; FlashInfer GDN prefill rejected; FA3
  `num_splits=1`; DeepGEMM scatter buffers zero-initialised.
- Batch-invariant aten overrides installed: `aten::mm`, `aten::addmm`,
  `aten::_log_softmax`, `aten::mean.dim`, `aten::rms_norm`, `aten::mm.dtype`,
  optionally `aten::bmm` ([`batch_invariant_ops/batch_invariant_ops.py:980-1002`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py#L980-L1002)).
- Piecewise/prefill CUDA graph auto-disabled
  ([`arg_groups/cuda_graph_hook.py:231-233`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/srt/arg_groups/cuda_graph_hook.py#L231-L233)).

Quantization: no code path rejects FP8/NVFP4 checkpoints in deterministic
mode, but the tracking issue lists blockwise FP8, per-channel FP8 GEMM, FP8
fused MoE and NVFP4 GEMM/MoE as **unchecked**. Treat quantized models as
unverified. CI-registered deterministic tests exist for BF16 dense
(FlashInfer/FA3/Triton/XPU), DeepSeek-V3, GLM-4-MoE-lite, Qwen3.5, Qwen3-Next,
and one `test_mla_fp8.py`
([`test/registered/attention/`](https://github.com/sgl-project/sglang/tree/0027af2e/test/registered/attention)).

---

## 7. Cross-machine / cross-SKU reproducibility

**Short answer:** SGLang makes **no** statement, and has **no** test, about
reproducibility across two machines of the same SKU or across SKUs (H100 PCIe
vs SXM, H100 vs H200, Hopper vs Blackwell). Every claim and every test is
about invariance to batch composition on one server. The nearest statements:

- Docs (`deterministic_inference.mdx`): "The same seed always produces the
  same response across different runs" — "runs" is not defined; the
  verification section runs a single server with varying batch sizes.
- Blog: benchmarks on "H100 80GB" and "H200 140GB" reported separately; no
  cross-device comparison; AMD support "with the extensibility of Triton
  backend".
- Tracking issue: "Not deterministic on Blackwell for TP4 #11513" (open) — an
  intra-SKU failure, indicating even same-hardware determinism is
  arch-dependent.
- The CI tests assert `Unique samples == 1` across batch sizes on one host
  ([`python/sglang/test/test_deterministic_utils.py:53-76`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/test/test_deterministic_utils.py#L53-L76);
  [`test_deterministic.py:2`](https://github.com/sgl-project/sglang/blob/0027af2e/python/sglang/test/test_deterministic.py#L2):
  "Batch the same prompt in random batch sizes, and test if the results are
  consistent across different trials.").
- Diffusion docs are candid that hardware changes numerics
  (`sglang-diffusion/performance-optimization.mdx:11`): "different kernels,
  GPU types, or precision paths can still introduce small numerical
  differences" — but this is not about the LLM deterministic mode.
- A grep across `docs/`, `python/sglang/test/`, `layers/sampler.py`,
  `murmur_hash.py`, and `batch_invariant_ops.py` for "cross-machine",
  "different GPU(s)", "across machines/hardware", "SXM", "PCIe", "H100 vs"
  finds nothing relevant.

Structural expectation (my analysis, not an SGLang claim): the RNG side is
hardware-independent (integer MurmurHash, float64 log/argmax), so the Gumbel
vectors will match across GPUs. The logits side is not guaranteed: attention
split sizes are keyed to backend defaults per arch (fa3 on Hopper vs flashinfer
on Blackwell), the deterministic default backend therefore *changes* between
SKUs, Triton kernels compile per arch, and cuBLAS (`torch` bf16 GEMM backend)
kernel selection depends on SM count and arch. Same-SKU/same-driver/same-wheel
machines are plausibly bit-identical but this is untested upstream.

---

## Practical notes for a verifier

1. Replay requires: model weights (hash them yourself), the `/server_info`
   dump (resolved args + `version`), and independently captured
   torch/CUDA/driver/flashinfer/sgl-kernel/triton versions plus GPU name.
   SGLang will not give you the latter.
2. Always send an explicit `sampling_seed`; treat omitted as `42`. Do not use
   `n>1` if you want distinct samples.
3. Avoid `min_p` with a seed (hard assert). Prefer no top-k/top-p if you want
   the sort-free path; otherwise accept the unstable-sort tie caveat.
4. Ask for `return_logprob=true` with `top_logprobs_num=<vocab>` or
   `token_ids_logprob=[0..V-1]` for full-row logprobs; they are temperature
   scaled `log_softmax` values unless `SGLANG_RETURN_ORIGINAL_LOGPROB=1`.
5. Check `meta_info.num_retractions` and `weight_version` on every response.
