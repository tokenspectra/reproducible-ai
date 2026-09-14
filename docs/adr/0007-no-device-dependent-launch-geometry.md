---
status: accepted
---
# Kernel launch geometry and reduction order depend on tensor shapes and dtype config only, never on the device

H100 PCIe and SXM share a compute capability but differ in SM count, clocks and bandwidth, and the same holds for other SKU pairs within an architecture. Bitwise identity across them, and therefore a single envelope per architecture, holds only if no kernel sizes its work by device properties. We forbid persistent "one program per SM" grids, occupancy-driven split-KV, timing-based autotuning, split-K that depends on M, floating-point atomics, and data-dependent early exits that change reduction order. Every custom kernel ships with a batch-invariance test across M in {1, 7, 64, 512, 1024}. Test 7 in the determinism matrix (same trace on H100 SXM and PCIe) enforces this.

## Consequences

Throughput is left on the table by design; overhead is measured and reported, not optimized away, until determinism is proven.
