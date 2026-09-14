---
status: accepted
---
# Determinism claim is bitwise logits on a fixed envelope, not cross-architecture

Bitwise identity across GPU architectures breaks on FMA contraction and tensor-core accumulation width, and across vendors it requires software-emulated arithmetic. We therefore claim bitwise-identical logits at every step only within an envelope (same GPU architecture, same compiled kernels, same dtype config, same weights), and make batch composition, restarts, chunking and prefill-vs-decode the axes that must not matter. A replay on a different envelope is refused by default and, when forced, reported as best effort with the first diverging step.

## Considered options

Same tokens only (hides the mechanism), same logits within architecture (chosen), same logits across architectures within a vendor (not achievable with native arithmetic), cross-vendor (emulation only).
