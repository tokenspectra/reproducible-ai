# reproducible-ai

Bitwise-deterministic LLM inference with self-contained, replayable traces that
an independent verifier can re-execute and confirm step by step.

- Design: [`docs/DESIGN.md`](docs/DESIGN.md)
- Demo spec: [`docs/DEMO.md`](docs/DEMO.md)
- Vocabulary: [`CONTEXT.md`](CONTEXT.md)
- Decisions: [`docs/adr/`](docs/adr/)
- Research: [`docs/research/`](docs/research/)

## Status

The PoC is built and runs end to end on live infrastructure:
[app.overversion.com](https://app.overversion.com/).

- `apps/web`: Next.js chat frontend with Google sign-in (Vercel).
- `services/api`: FastAPI service for conversations, traces, resume tokens and
  verify (Railway).
- `packages/core`: the `reproducible_ai` trace library (hashing, seeds, signed
  tokens, evidence).
- `cloud/nebius`: one H100 SXM VM running SGLang in deterministic mode with
  Qwen3-8B, behind `infer.overversion.com`.

What it showed: bit-exact inference recompute on one GPU with SGLang's
deterministic mode. A probe request with 1–12 concurrent neighbours gave 1
unique output in deterministic mode and 6 of 6 different in the baseline
(`scripts/chaos_check.py`). The same seed gives the same output across forks
and accounts, including the streaming path. The run envelope is in
`cloud/nebius/envelopes/`.

What it does not cover: more than one GPU or server process, restarts, other
engines (vLLM) or GPUs, prefix cache on, quantised or MoE models, measured
overhead, and training recompute.

**The H100 box is up only for demo sessions** (it costs $3.85/h; see
[`cloud/nebius/README.md`](cloud/nebius/README.md)). When it is down,
`infer.overversion.com` does not answer: that is expected, not broken. The app
stays usable and falls back to OpenRouter, with those answers labelled "no
guarantee".

## Relation to TokenSpectra

This is TokenSpectra's earlier PoC. Its single-GPU recompute is what
[TokenSpectra](https://github.com/tokenspectra) shows as its recompute engine;
deterministic inference is not the default in today's datacenters, and
TokenSpectra's next step is checking bit-exactness across two GPUs.

License: MIT.
