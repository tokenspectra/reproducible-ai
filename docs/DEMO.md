# overversion.com demo spec

Status: agreed 2026-09-14. Deadline: 2026-09-23 (promise made in the ComidasLEX
group on 2026-09-09: "en 2 semanas top voy a hacer lo de las salidas idénticas").
Audience: Guillem first, then anyone with a Google account.

The demo shows the one thing that was promised: fork a conversation, send the
same message to both copies, get byte-identical answers, and hand someone a
token that lets them reproduce your exact state and continue from it.

## Hosts

| Host | What | Where |
|---|---|---|
| `app.overversion.com` | Next.js chat frontend, Google sign-in (Auth.js) | Vercel |
| `overversion.com` | redirect to `app.overversion.com` (a landing page is out of scope for the demo) | Vercel |
| `api.overversion.com` | FastAPI service: users, conversations, traces, tokens, verify, upstream status | Railway, always on |
| `infer.overversion.com` | SGLang deterministic mode, Qwen3-8B, plus a baseline instance at `/baseline/` | Nebius H100 SXM, up only for sessions (`cloud/nebius/`) |
| OpenRouter `qwen/qwen3-8b` | always-available non-reproducible engine, and what the app falls back to when the box is down | external |

The Python service owns Postgres and all trace logic (it is the repo's
`reproducible_ai` library behind HTTP). The frontend does auth and rendering
only and calls the API with the user's identity in a signed header.

## UI

One chat panel. Left navbar: the user's conversations, newest first, with a
"new" button and, on each row, a fork icon.

**Conversation header**: title, engine badge for the last turn, envelope id,
the resume token (truncated, copy button), "verify" button, and when the
conversation is a fork, "identical to parent" or "diverged at token N" once
both have answered the same latest message.

**Composer**: text box, send. **Below it, a mode selector** with three engines.
The default is Reproducible whenever the box is up, otherwise OpenRouter.

| Option | Backend | Evidence |
|---|---|---|
| **Reproducible** | SGLang deterministic mode on `infer.overversion.com` | full: seed, envelope, per-step hashes, token |
| **Standard** | the SGLang baseline instance, same model and weights, no determinism flag | none; exists so the comparison is apples to apples |
| **OpenRouter** | `qwen/qwen3-8b` via OpenRouter, always available | none; the seed is sent but nothing is guaranteed |

Reproducible and Standard are greyed out with a tooltip when the box is down,
and the selector falls back to OpenRouter. OpenRouter is always selectable, so
the app is usable with the GPU off; the pill and the turn header make clear
that those answers carry no guarantee.

The selector applies to the next message. Every assistant turn records the
engine that produced it, so a conversation may mix engines; only reproducible
turns carry evidence and take part in the identical/diverged comparison. A
fork of a conversation whose last turn was not reproducible still works: the
fork starts reproducible from the next turn, and the comparison begins there.

**Status pill** in the top bar: `infer.overversion.com` up or down, model,
deterministic flag, envelope id, last checked. Polled by the API every 10 s
and pushed to the UI.

**New conversation dialog**: title (optional), and one optional field that
accepts either a resume token (start from someone's exact state) or a plain
seed (start fresh with a chosen seed, recorded as externally supplied). Blank
means a fresh OS-random root seed.

**Verify view**: replays the conversation's reproducible turns by one prefill
of the full transcript with logprobs at prompt positions, and renders a tick
or a cross per token, with the first divergence highlighted.

**Replay-N panel** (stretch): re-sends the last user message N times
concurrently under background load in the selected mode and shows the number
of unique outputs. Reproducible shows 1; standard usually does not.

## Resume token

Signed (Ed25519, key held by the API), transparent (base64url of a small JSON
payload), a few hundred bytes:

```
v, trace_id, turn, pos, model_id, envelope_id, root_seed (or encrypted),
prefix_hash (BLAKE3 of rendered prefix token ids), evidence_hash (last step)
```

Setting a token on a new conversation: verify the signature, load the source
trace, copy its messages as the prefix, adopt root seed and turn index, and
verify the first response's evidence before streaming it ("state
reproduced"). Possession of a token grants read access to its prefix; this is
stated in the dialog. Works across users.

## Fork

Creates a new conversation with the messages copied, the same root seed and
turn index, `forked_from` set. The navbar shows it under the parent.

## Evidence

Per reproducible step: token id and BLAKE3-128 of the top-32 logprobs from
SGLang, recorded as `logits_hash_kind = "top32_logprobs_scaled"`. Per turn: the
envelope id from `/server_info` plus the saved envelope JSON, the seed used,
sampling params. Every request to SGLang sets an explicit `seed`
(SGLang defaults to 42 and does not vary it for `n > 1`).

## Status and fallbacks

- Box up: all three engines selectable, Reproducible preselected.
- Box down: Reproducible and Standard disabled with a tooltip, OpenRouter preselected.
- OpenRouter also down: send disabled with the reason on the button.
- Any turn not produced in Reproducible mode is labelled "no guarantee" in its header and carries no evidence or token.

## Out of scope for the demo

Tool calls, file uploads, sharing controls beyond the token, billing, rate
limits beyond a per-user daily cap, the local reference oracle as a backend.

## Build order

1. API skeleton on Railway: Postgres schema (users, conversations, turns,
   steps, settings), upstream status poller, OpenRouter client, SGLang client
   with seed and logprobs, SSE streaming.
2. Trace core in `reproducible_ai`: hashing, KDF, token sign/verify, evidence.
3. Frontend on Vercel: Auth.js Google, navbar, chat, selector, status pill,
   new-conversation dialog, fork.
4. Verify view, then the parent/fork comparison, then replay-N.
5. Wizard for the accounts and secrets (Google OAuth client, Vercel, Railway,
   OpenRouter key, DNS records), Nebius box per `cloud/nebius/README.md`.
