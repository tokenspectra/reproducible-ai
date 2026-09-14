---
status: accepted
---
# Per-step evidence is a BLAKE3-128 hash of the fp32, pre-temperature logits

Evidence must be cheap, catch any divergence at the exact step, and be comparable across traces forever, so the definition is fixed: the full-vocabulary fp32 logits as produced by the lm_head accumulator, before temperature or any cast, serialized little-endian in vocabulary order, hashed with BLAKE3 and truncated to 128 bits. Hashing before the output cast is what keeps stochastic rounding of that cast observable. Where an engine cannot expose raw logits (SGLang returns temperature-scaled log-softmax), the step record's `logits_hash_kind` names the quantity so a verifier compares like with like. Debug level adds per-layer hidden-state hashes and an optional full-logits sidecar.

## Considered options

bf16-cast logits (loses information, hides SR). Probabilities (temperature-dependent). xxh3 (non-cryptographic; BLAKE3 is fast enough and keeps the integrity story consistent). SHA-256 (slower, no benefit).
