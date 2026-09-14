---
status: accepted
---
# Local development pins torch 2.9.1+rocm6.4 with Triton 3.5.1 on the RX 6700 XT run as gfx1030

The local GPU is natively gfx1031 (RDNA2) and is unsupported by every serving engine; it works with PyTorch ROCm and Triton only under `HSA_OVERRIDE_GFX_VERSION=10.3.0`. Triton 3.5.1 (bundled with torch 2.9.1) compiles and runs fp32/fp16/bf16 `tl.dot` kernels correctly and bitwise-stably on it; Triton 3.6 through 3.8 miscompile bf16 elementwise ops on gfx10.3 (`v_dot2_bf16_bf16` selection failure, fixed only after 3.8). Torch's flash and memory-efficient attention ship no gfx1030 binaries, so any attention kernel used locally must be our own or the math path. RDNA2 has 64 KB LDS, wave32 and no matrix cores, so tiles are small and fp16 is about three times faster than bf16. Do not upgrade the local venv without re-running the kernel test suite.

## Consequences

The local machine is a correctness lab, not a performance reference. Cloud runs use a separately pinned container image.
