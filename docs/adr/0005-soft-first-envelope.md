---
status: accepted
---
# Envelope fields are soft by default; six are hard; the kernel field hashes compiled binaries

We do not know a priori which environment facts flip bits, so asserting a long hard list would be guesswork that makes verification brittle. Only six fields refuse a verify on mismatch: in-memory weights hash, tokenizer plus chat-template hash, dtype config, kernel registry hash, GPU architecture, parallel config. The kernel registry hash covers the compiled device binaries of every custom kernel, which subsumes Triton version, compiler flags and target in one field, so those become soft. All other facts (torch version, driver, runtime, BLAS versions, SKU string, SM count, git SHA, hostname) are recorded and compared but never block. A soft field is promoted to hard only when a test in the determinism matrix shows it can change a bit, and the promotion is recorded as a finding. The expected first promotion is the torch version, because elementwise ops stay in torch.

## Consequences

`gpu_arch` is the hard field, not the SKU, so H100 PCIe and SXM share an envelope; this is what ADR-0007 protects.
