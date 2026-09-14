---
status: accepted
---
# Random draws are keyed by request-local element identity, never by batch location

If a stochastic-rounding mask or a sampler draw were keyed by an element's row in the batched tensor, the mask would change with batch composition and determinism would be lost even with invariant kernels. All draws use Philox4x32-10 with an explicit key/counter layout: stochastic rounding is keyed by (SR seed, tensor id, absolute position, feature index) and sampling by (sampling seed, absolute position, draw index). Absolute token position replaces any "timestep" and makes prefill and decode produce identical masks and draws for the same token, which is also what fast resume relies on. Tensor ids come from the model graph, never from pointers or allocation order. Seeds derive from one root seed per session (turn seed, then sampling and SR subkeys), each recorded explicitly and independently overridable; the root seed is OS-random by default and may be supplied externally, with the source recorded.

## Considered options

Threefry (no advantage, no Triton support). Per-thread generator streams (scheduling-dependent). Independent unrelated seeds for sampling and SR (more flags for no benefit; derived subkeys with overrides give the same experiments).
