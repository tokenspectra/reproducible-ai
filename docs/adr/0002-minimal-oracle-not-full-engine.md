---
status: accepted
---
# A minimal plain-torch oracle plus engine adapters, not our own batch-invariant bf16 engine

SGLang's deterministic mode and Thinking Machines' `batch_invariant_ops` already provide batch-invariant bf16 GEMM, RMSNorm, softmax and fixed-split attention, so writing our own would be redundant work. What does not exist is the trace format, the verifier, the envelope characterization, prefill-vs-decode and resume guarantees, and determinism for stochastic rounding and FP8/NVFP4. We keep a minimal reference oracle in plain torch, locally runnable on the AMD card where SGLang and vLLM cannot run, purely to make the format and verifier testable end to end; custom kernels are written only where prior art has none (SR epilogue, quantizers, stable-tie sampler); the credibility result comes from adapters (SGLang first, vLLM later). A short SGLang characterization spike on one H100 precedes the kernel phase and decides how much of it is needed.

## Considered options

Extend vLLM only (no local iteration possible on gfx1030). Full standalone engine with own kernels (redundant with prior art). tinygrad as substrate (rejected: its AMD backend refuses gfx10, and its kernel heuristics include the batch dimension in the AST, so kernels and bits change with batch size unless optimizations are disabled). Chosen: minimal oracle plus adapters.
