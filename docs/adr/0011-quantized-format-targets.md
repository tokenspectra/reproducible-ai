---
status: accepted
---
# Cloud quantization targets are FP8 W8A8 on H100 and NVFP4 W4A4 on B200; weight-only int4 is out of scope

The brief asks for what serving fleets actually run. Weight-only int4 (AWQ/GPTQ) dequantizes to bf16 and computes in bf16, so it teaches nothing about determinism. FP8 with e4m3, per-token dynamic activation scales and fp32 accumulation on H100, and NVFP4 (e2m1 values, e4m3 block scales over 16, fp32 tensor scale) on B200, both with FP8 KV cache, put a dynamic quantizer in the path, which is exactly where a fixed-order amax reduction and seeded stochastic rounding matter. SM120 Blackwell cards (RTX PRO 6000, RTX 5090) run different NVFP4 kernels from SM100 and are never a determinism claim. Locally both tiers are emulated in software to test the quantizer and the rounding mask bit-for-bit. The open risk is that the cloud FP8/NVFP4 GEMMs themselves come from libraries whose reduction order we do not control.
