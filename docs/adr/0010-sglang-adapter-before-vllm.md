---
status: accepted
---
# The first engine adapter targets SGLang, vLLM second

SGLang's deterministic mode already has a seeded counter-hash sampler, fixed-split attention on three backends, a pinned all-reduce on NVIDIA and AMD, and per-op coverage close to our kernel registry, so a trace writer and verifier backend attach with less engine surgery. vLLM's mode leaves the sampler unchanged and its GEMM overrides are CUDA-only. vLLM follows as a second adapter because supporting two engines is itself evidence that the trace format is engine-neutral. Adapter caveats found in the audit: SGLang defaults the seed to 42 and does not bump it for `n > 1`, rejects `min_p` with a seed, chooses its deterministic attention backend by GPU architecture, and exposes full-vocabulary evidence only as scaled log-softmax.
