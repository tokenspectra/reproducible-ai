# Local GPU feasibility: RX 6700 XT (gfx1030/gfx1031, RDNA2) + ROCm 6.4.1 for bitwise-deterministic LLM inference

Research date: 2026-09-09. Sources are primary (GitHub source files, issues, PRs, release notes, official docs) plus experiments run on this machine. Every third-party claim carries a link; claims that could not be confirmed are marked **unverified**.

Local stack used for the experiments: `torch 2.9.1+rocm6.4` (HIP 6.4.43484), `pytorch-triton-rocm 3.5.1`, ROCm at `/opt/rocm-6.4.1`, `HSA_OVERRIDE_GFX_VERSION=10.3.0` already set in the environment. `torch.cuda.get_device_properties(0)` reports `AMD Radeon RX 6700 XT`, `gcnArchName='gfx1030'`, `major=10, minor=3`, `warp_size=32`, 12272 MB, 20 CUs. Note: the KFD-native target of this card is **gfx1031** (tinygrad's `/dev/kfd` path reported `Unsupported arch: gfx1031`); ROCm libraries ship only gfx1030 code objects, which is why the override is needed. gfx1030 and gfx1031 are ISA-identical within LLVM's `GK_GFX1030..GK_GFX1036` range.

## Executive summary

1. **SGLang: not feasible on this GPU.** The ROCm kernel build (`python/sglang/kernels/aot/setup_rocm.py`) hard-exits for anything but `gfx942 / gfx950 / gfx1250`; PyPI ships no ROCm kernel wheel at all; the Docker images are MI300/MI350-only on ROCm 7.x; the HIP AOT kernels hard-code `WARP_SIZE 64` on the host side (breaks wave32 RDNA); there are zero RDNA2 reports in the tracker and even RDNA3 (7900 XTX) is still "blocked" in the official consumer-Radeon tracking issue #30599, with an unresolved decode hang. `--enable-deterministic-inference` itself is Triton-based, has an explicit AMD path (fixed 256-token decode split tiles, 1-stage extend kernel, vendored batch-invariant Triton GEMMs, seeded Gumbel sampling, deterministic 1-stage HIP all-reduce) and is CI-tested on MI300, but only behind an engine that cannot be built for gfx1030.
2. **vLLM: buildable from source, unsupported, and its batch-invariant mode does nothing for GEMMs on ROCm.** gfx1030 is in `CMakeLists.txt` `HIP_SUPPORTED_ARCHS` but not in the docs, wheels or Docker arch lists; `vllm/platforms/rocm.py` classifies gfx1030 as neither `_ON_GFX1X` nor `_ON_CDNA`, so RDNA-specific paths silently turn off (open PR #52391 fixes this). People run source builds on RX 6800/V620 (30-60 tok/s fp16 GPTQ at v0.21, regressed to 1-2 tok/s in v0.22; bf16 is software-emulated on RDNA2 and collapses throughput). `VLLM_BATCH_INVARIANT=1` installs the Triton `mm/addmm/matmul/linear` overrides only under `is_cuda()` (SM80 family) or `is_xpu()`; on ROCm it installs only the softmax/mean/bmm/rms_norm overrides and leaves GEMMs on hipBLAS/hipBLASLt. The only ROCm-usable backend with `supports_batch_invariance() == True` is `TRITON_ATTN`. The AMD PR #52231 adding ROCm support is open, CDNA-only tested.
3. **Triton on gfx1030: works, and is the viable kernel substrate.** Triton recognises gfx1030 as `ISAFamily::RDNA2` (wave32, 64 KB LDS), lowers `tl.dot` to the FMA path (`v_dot2_f32_f16` for f16, fp32 FMA for bf16/f32, no MFMA/WMMA), and never runs gfx10 in CI. Locally, Triton 3.5.1 compiled and ran fp16/bf16/fp32 `tl.dot` matmuls and `tl.sum` reductions correctly, bitwise-stable run-to-run and batch-invariant for a fixed-tile kernel; `torch.compile`/Inductor also works. The upstream TML `batch_invariant_ops` kernels run unmodified for bf16/fp16 and are batch-invariant here; the fp32 config exceeds 64 KB LDS unless `num_stages` is reduced. Avoid Triton 3.6.0-3.8.0 on gfx1030: they lower any bf16 elementwise multiply on RDNA to `v_dot2_bf16_bf16`, which gfx10.3 lacks (`LLVM ERROR: Cannot select`), fixed on main by PR #11227 after the 3.8.0 cut. Triton 3.5.1 (the version pinned by the last ROCm 6.4 PyTorch wheels) is unaffected.
4. **tinygrad: not a sensible substrate here.** Its AMD backend (both KFD and the AM driver) asserts `Unsupported arch` for gfx10 since PR #16017 (2026-05-02; releases 0.13/0.14) and has no gfx10 register autogen at all; the maintainer declined a working RDNA2 PR (#17666). `DEV=CL` (ROCm OpenCL) runs on this card; `DEV=HIP` runs only with `TC=0` because `tc.get_amd()` defaults unknown archs to RDNA3 WMMA intrinsics. Kernel shapes are chosen deterministically per kernel AST, but the AST includes the batch dimension and the hand-coded heuristics (GROUP/GROUPTOP split-K, unroll, upcast) change with it: locally, `[M,4096]@[4096,512]` picked three different kernels for M=1/7/256 and row 0 differed bitwise between M=1 and M=256; `NOOPT=1` restores invariance but is unusably slow. BEAM is wall-clock based. The pinning hook (`KernelInfo.opts_to_apply`) has no public API. No continuous batching or paged KV exists. PyTorch + a few custom Triton kernels with tiling fixed per (K, dtype) is the better path on this machine.
5. **Batch-invariant reference implementations.** TML `batch_invariant_ops` is MIT-licensed, pure Triton, provides persistent `mm/addmm`, `log_softmax`, `mean.dim` only (no attention kernel; it returns a 16x16 FlexAttention block size), installed via `torch.library.Library("aten","IMPL")` with the dispatch key of the current accelerator (resolves to `"CUDA"` on ROCm builds too). vLLM's mode additionally covers `rms_norm`, `bmm`, `softmax`, linear/LM-head, fixed MoE Triton config, pinned NCCL tree all-reduce, TF32 off; attention via FLASH_ATTN (num_splits=1), TRITON_ATTN, FLEX (16x16), TRITON_MLA; CUDA graphs/torch.compile supported; prefix caching, spec decode, GDN/Mamba, FlashInfer (issue #2424) are gaps; sampling is not modified (relies on per-request seed). SGLang's mode covers mm/addmm/bmm/log_softmax/mean/rms_norm (vendored TML fork, optional DeepGEMM), attention via FlashInfer (fixed 2048/4096 split tiles, no radix cache), FA3/FA4 (num_splits=1), Triton (256-token split tile, unified extend kernel, radix cache kept with 4096-aligned chunk truncation), seeded Gumbel-max sampling with murmur hash, fixed MoE config, deterministic AMD 1-stage all-reduce; gaps: EP, speculative decoding, Blackwell TP4 (#11513), FlashInfer + radix cache.

**Practical recommendation:** on this machine, build the deterministic engine on `torch 2.9.1+rocm6.4` + Triton 3.5.1 custom kernels (fixed tiles per K/dtype, no split-K, no atomics, fixed split-KV *size* for attention, `num_stages` sized for 64 KB LDS), fp16 rather than bf16 for throughput, with an own batching layer. Use TML/vLLM/SGLang kernels as design references, not as engines. Do not expect matrix-core throughput: everything is FMA on RDNA2.

---

## Local experiments (this machine, 2026-09-09)

Scripts were run from the project `.venv` (torch 2.9.1+rocm6.4, Triton 3.5.1) unless noted. Reference results computed in fp64.

### Triton 3.5.1 on gfx1030

A plain 64x64x32-tile `tl.dot` matmul kernel (fp32 accumulator, no split-K) and a `tl.sum` row-reduction kernel:

| Test | Result |
|---|---|
| `tl.dot` fp16 in / fp32 acc, [256,4096]x[4096,512] | compiled in 0.6 s, max err vs fp64 5.8e-4, bitwise-equal over 5 reruns, row 0 identical for M=1 vs M=256 |
| `tl.dot` bf16 | compiled, max err 4.6e-4, bitwise-stable, batch-invariant |
| `tl.dot` fp32 | compiled, max err 7.8e-4, bitwise-stable, batch-invariant |
| `tl.sum` over 8192 fp32 | max err 3.6e-5 vs fp64, bitwise-stable |
| `torch.compile` (Inductor) of RMSNorm and SiLU-MLP in bf16 | compiled and ran; Inductor warned `GPU arch does not support max_autotune_gemm mode usage` (Inductor's `is_big_gpu()` returns False for `prop.major == 10` on HIP, `torch/_inductor/utils.py`) |

The installed Triton HIP backend (`triton/backends/amd/compiler.py`) sets `warp_size = 32 if gfx_major >= 10 else 64`, applies `'-real-true16'` only for gfx11, and gates in-thread-transpose/async-copy/pingpong to gfx942/gfx950/gfx11xx/gfx12xx, so gfx1030 gets the plain pipeline.

### torch.mm (rocBLAS/hipBLASLt) is not batch-invariant on this GPU

Comparing row 0 of `a[:M] @ b` against row 0 of the M=2048 result (`preferred_blas_library = Cublas`, i.e. hipBLAS):

| dtype, K | M values where row 0 differs bitwise | run-to-run |
|---|---|---|
| fp16, K=4096 | {512} | equal |
| fp16, K=11008 | {1, 7, 64, 512, 1024} | equal |
| bf16, K=4096 / 11008 | none observed | equal |
| fp32, K=4096 | {64, 1024} | equal |
| fp32, K=11008 | none observed | equal |

This is exactly the failure mode described by the TML blog: deterministic run-to-run, but the result of a row depends on how many other rows are in the batch.

### Upstream TML `batch_invariant_ops` on gfx1030 (commit f22b1fb, 2025-11-03)

Cloned and imported unmodified with the project venv:

| Config | bf16 | fp16 | fp32 |
|---|---|---|---|
| as shipped (`num_stages=3`) | OK: row 0 identical for M in {1,7,64,512,1024} vs 2048; bitwise run-to-run; `log_softmax` row-invariant; `mean` works | OK (same) | **FAIL** `OutOfResources: out of resource: shared memory, Required: 81920, Hardware limit: 65536` |
| `num_stages` 3 -> 1 | OK | OK | OK, max err vs fp64 1.06e-3 |

(bf16/fp16 max error vs fp64 was ~1.0 / ~0.125 at output magnitude ~64, i.e. 1-2 ulp of the *output* dtype; accumulation is fp32.) The `mean` override also hijacks fp64 `mm` (a `KeyError: torch.float64` if you compute a fp64 reference inside the mode), so compute references outside `set_batch_invariant_mode`.

### tinygrad 0.14.0 on this card (scratch venv, `pip install tinygrad`)

| Backend | Result |
|---|---|
| `DEV=AMD` (KFD `/dev/kfd`) | `AssertionError: Unsupported arch: gfx1031` from `tinygrad/runtime/ops_amd.py:1003`: `assert (self.target in ((9,4,2),(9,5,0))) or self.target[0] in (11, 12)` |
| `DEV=AMD` with that assert patched to allow 10 | `AttributeError: no such autogen: soc_10` (`runtime/support/amd.py: import_soc`) - there are no gfx10 register/soc tables in the package |
| `DEV=HIP` (needs `ROCM_PATH=/opt/rocm-6.4.1`, not `HIP_PATH`) | `CompileError`: the HIPRenderer emitted `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` because `codegen/opt/tc.py: get_amd(arch)` is `{"gfx942":..., "gfx950":..., "gfx1200":..., "gfx1201":...}.get(arch, amd_rdna3)` - any unknown arch, including gfx1030, gets RDNA3 WMMA. With `TC=0` it runs. |
| `DEV=CL` (ROCm OpenCL ICD `amdocl64_60401_83.icd`) | runs |
| `DEV=GPU` | no such backend any more (`ops_gpu` removed; it is `CL`) |

Batch invariance with default hand-coded optimisations (`DEV=CL` and `DEV=HIP TC=0`, fp16 inputs, fp32 accumulate, `[M,4096]@[4096,512]`):

| M | kernel chosen | row 0 equal to M=256 result? |
|---|---|---|
| 1 | `r_512_16_256`, opts `(GROUPTOP axis=0 arg=16)` - 16-lane split-K over 256 each, two-stage reduce through LDS | **no** |
| 7 | `r_7_8_16_4_1024_...` | yes (by luck of the same K order) |
| 256 | `r_8_8_8_16_4_...` | - |

With `NOOPT=1` all three agree bitwise (plain sequential K loop). Also noted: `Tensor.matmul(dtype=float32)` on fp16 inputs multiplies in half and casts afterwards - the generated kernel contains `((float)((val0*val1)))` - giving ~0.05 max error vs fp64 instead of ~6e-4 for a true fp32-accumulate fp16 GEMM.

---

## 1. SGLang on ROCm

### 1.1 Officially supported archs

- Official docs (`docs/docs/hardware-platforms/amd_gpu.mdx`, rendered at https://docs.sglang.io/platforms/amd_gpu.html) mention only Instinct ("such as MI300X"; "MXFP4 requires CDNA3/CDNA4"). No Radeon/RDNA/gfx10xx/gfx11xx anywhere.
- `docker/rocm.Dockerfile` (https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile): build targets are `gfx942*`, `gfx950*`, `gfx1250-rocm1000`; `ARG ROCM_DEVICE_ARCH_LIST="gfx942 gfx950 gfx1250"`; `*) echo "Unsupported ... GPU_ARCH=${GPU_ARCH}"; exit 1`. Base images are ROCm 7.x (`rocm/pytorch:rocm7.2*`), not 6.4; `BUILD_AITER_ALL="1"`.
- sgl-kernel was merged into the main tree on 2026-07-29 (PR #32648); the ROCm build file is now `python/sglang/kernels/aot/setup_rocm.py` (https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/aot/setup_rocm.py):

  ```python
  default_target = "gfx942"
  amdgpu_target = os.environ.get("AMDGPU_TARGET", default_target)
  if torch.cuda.is_available():
      amdgpu_target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
  ...
  if amdgpu_target not in ["gfx942", "gfx950", "gfx1250"]:
      print(f"Warning: Unsupported GPU architecture detected '{amdgpu_target}'. Expected 'gfx942', 'gfx950', or 'gfx1250'.")
      sys.exit(1)
  ```
  So **sgl-kernel does not build for gfx1030**: it auto-detects the local GPU and aborts. The only gfx12 target is gfx1250 (Instinct MI450, not consumer RDNA4), added in PRs #32466 (2026-08-06) and #36871 (2026-08-31). The legacy `3rdparty/amd/wheel/sgl-kernel/CMakeLists_rocm.txt` is stricter (`FATAL_ERROR ... Expected 'gfx942' or 'gfx950'`).
- Wave-size hard-code that breaks RDNA even if the allowlist is patched: `python/sglang/kernels/aot/include/utils.h` defines `WARP_SIZE 64` whenever `__GFX9__` is defined *or* `__HIP_DEVICE_COMPILE__` is undefined (the host pass), so launch geometry mismatches `__launch_bounds__` on wave32 parts -> `hipErrorLaunchFailure` (documented in PR #31803, https://github.com/sgl-project/sglang/pull/31803). Affected: `csrc/moe/moe_align_kernel.cu`, `moe_topk_softmax_kernels.cu`, `moe_topk_sigmoid_kernels.cu`, `allreduce/custom_all_reduce_hip.cuh`, GGUF mmq/mmvq.
- No ROCm wheel exists: PyPI `sglang-kernel==0.4.6.post1` (pinned in `python/pyproject.toml`) has only two CUDA manylinux files; zero files containing "rocm"/"hip" across all `sglang-kernel` and legacy `sgl-kernel` releases (https://pypi.org/pypi/sglang-kernel/json, https://pypi.org/pypi/sgl-kernel/json).
- PR #30415 "Enable RDNA3/4 (gfx1100/gfx1201) for ROCm kernels" merged 2026-07-07 into the **`amd_march` branch, not `main`** (https://github.com/sgl-project/sglang/pull/30415); maintainer HaiShaw: "this is very limited support. We will not have CI to cover it yet."
- MFMA/WMMA: the HIP sources in the `setup_rocm.py` list (allreduce, activation, topk, moe_align, rotary, kv-transfer, eagle utils) are elementwise/reduction kernels; RDNA3 users report "all 14 HIP compile units succeed" on gfx1101 after patching the allowlist (issue #27519). No direct MFMA/WMMA use found in that list; **unverified for RDNA2** (nobody has compiled it for gfx103x). The CDNA-only pieces are the custom/quick/deterministic all-reduce collectives, which the RDNA PRs guard out with `-DSGL_IS_RDNA` (PR #32543).

### 1.2 Community reports on RDNA2 / RDNA3

- **RDNA2: no reports at all.** Issue search over `sgl-project/sglang` for `gfx1030`, `RDNA2`, `6700 XT`, `6800 XT`, `6900 XT` returns no user reports. PR #20925 "Add Radeon i8090s (gfx1030) unified memory support" (2026-03-19, closed unmerged) describes a non-existent GPU and should be disregarded. The closest RDNA2 data point is ROCm/aiter issue #210 (2025-03-17) asking whether the Triton MHA kernel supports gfx1030 - open, no reply (https://github.com/ROCm/aiter/issues/210); aiter's README lists gfx942/gfx950 supported, gfx1100/1151/1201 experimental, no RDNA2.
- **RDNA3: works with patches, never officially:**
  - #3243 (2025-01-31): RX 7900 XTX, ROCm 6.2.4, vLLM-derived `rocm/vllm-dev:navi_base` image, `--attention-backend triton`, ~28 tok/s Llama-3.1-8B decode; later users could not reproduce on newer versions (https://github.com/sgl-project/sglang/issues/3243).
  - #27519 (2026-06-08): RX 7800 XT (gfx1101), ROCm 7.2.4, v0.5.10.post1; sgl-kernel builds with a 1-line allowlist patch; served with `--attention-backend torch_native --disable-cuda-graph --disable-radix-cache`; blockers: CUDA flashinfer wheel must be uninstalled, `apache-tvm-ffi` CUDA addon, "`import aiter` at import time in ~36 files is fatal on a box without aiter" (https://github.com/sgl-project/sglang/issues/27519).
  - #30245 (2026-07-06): 4x W7900, MoE page faults because the installed `sgl_kernel` embedded only gfx942 code objects; "The triton fused-MoE kernel is correct on gfx1100"; needs `--disable-cuda-graph` (https://github.com/sgl-project/sglang/issues/30245).
  - #30599 (2026-07-09, **open**) "Officially support consumer Radeon RDNA3/RDNA4": gfx1100/1101/1200/1201/1150/1151 all "blocked"; as of 2026-07-17 the server boots on gfx1100 but decode hangs ("120s hang; GPU 100% use... no `Decode batch` line") with both triton and torch_native backends (https://github.com/sgl-project/sglang/issues/30599). RDNA2 is outside even this issue's scope.
  - Open unmerged RDNA PRs: #13975 (gfx1201), #28518/#31137 (gfx1151), #31803/#32543 (wave32 `WARP_SIZE`), #32092 (experimental gfx1201, Triton attention). None add gfx103x.

### 1.3 `--enable-deterministic-inference`

Source locations (post-refactor `main`, HEAD 0027af2, 2026-09-09):

- Flag: `python/sglang/srt/arg_groups/fields/exec_.py` (`enable_deterministic_inference: bool = False`, "Enable deterministic inference mode with batch invariant ops"); env mirror `SGLANG_ENABLE_DETERMINISTIC_INFERENCE`.
- Allowed backends: `python/sglang/srt/arg_groups/choices.py`: `DETERMINISTIC_ATTENTION_BACKEND_CHOICES = ["ascend", "fa3", "fa4", "flashinfer", "intel_xpu", "triton"]`; `RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND = ["ascend", "fa3", "fa4", "triton"]`. Any other backend raises `ValueError`. If unset, the default is chosen by NVIDIA SM level (`arg_groups/overrides.py`), so on ROCm you must pass `--attention-backend triton`.
- Docs (`docs/docs/advanced_features/deterministic_inference.mdx`, https://docs.sglang.io/advanced_features/deterministic_inference.html): "only supported with the following three attention backends: FlashInfer, FlashAttention 3 (FA3), and Triton"; table: FlashInfer = CUDA graph yes / chunked prefill yes / radix cache **no** / temperature>0 yes; FA3 and Triton = all yes.
- Blog (2025-09-22, https://lmsys.org/blog/2025-09-22-sglang-deterministic/): "Triton - Fixed decoding split size; supports AMD hardware"; overhead 25-45% (34.35% avg for FlashInfer/FA3, Triton +44-55% latency). Shipped in v0.5.3 (https://github.com/sgl-project/sglang/releases/tag/v0.5.3), Triton PR #10694 "(Hardware test: NV and AMD GPUs)" with MI300 results (https://github.com/sgl-project/sglang/pull/10694). Tracking issue #10278 (https://github.com/sgl-project/sglang/issues/10278).

What it does:

- **Batch-invariant ops**: `model_runner.maybe_enable_batch_invariant_mode()` -> `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`, a **vendored fork** of TML's package ("Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops"), not a pip dependency. Overrides `aten::mm`, `addmm`, `_log_softmax`, `mean.dim`, `rms_norm`, `mm.dtype`, optionally `bmm` via `torch.library.Library("aten","IMPL")`; dispatch key `"CUDA"` when `is_cuda_alike()` (CUDA **or HIP**), `"XPU"`, `"NPU"`. Persistent Triton GEMM grid sized by `multi_processor_count` (works on HIP); optional DeepGEMM for bf16 (`SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM`, CUDA-only) with Triton fallback.
- **Triton attention** (`layers/attention/triton_backend.py`): `split_tile_size = SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE` (default 256), `num_kv_splits = ceil(seq_len / split_tile_size)` per sequence; extend uses `_forward_extend_unified` (1-stage kernel) instead of the 2-stage kernel.
- **FlashInfer**: `prefill_split_tile_size` 4096 / `decode_split_tile_size` 2048, split-KV disabled under CUDA graph; relies on FlashInfer PR #1675 (FA2 paths only). Not available on ROCm ("FlashInfer is not available on Non-NV platforms").
- **FA3/FA4**: `num_splits = 1`; NVIDIA SM80/90/100 only.
- **Sampling**: forces `sampling_backend = "pytorch"`; `layers/sampler.py: multinomial_with_seed` = Gumbel-max with `murmur_hash32(seed, position, col)` (a Triton kernel), fp64 noise; default `sampling_seed=42`. Caveat in code: with top-k/top-p the sorted probs must be renormalised or results are wrong; bugs #15481/#15746 (seed not forwarded on OpenAI endpoints).
- **Radix cache**: kept for fa3/fa4/triton; chunked-prefill truncation aligned to `SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE` (4096) (`managers/scheduler.py`, `schedule_policy.py`); disabled for FlashInfer.
- **CUDA graphs**: supported ("2.8x acceleration" over eager deterministic).
- **MoE**: `fused_moe_triton_config.py` pins `BLOCK_SIZE_M=64, N=64, K=32, GROUP_SIZE_M=8`; `cutedsl` BF16 GEMM forbidden; DeepEP v2 deterministic sorting unsupported.
- **TP**: CUDA pins `SGLANG_DETERMINISTIC_NCCL_NCHANNELS=8`; **AMD** uses a 1-stage deterministic HIP all-reduce ("fixed accumulation order, no atomics", PR #15340 merged 2025-12-19, MI350/MI355 tested, https://github.com/sgl-project/sglang/pull/15340) via `SGLANG_USE_1STAGE_ALLREDUCE` - this is exactly the collective the RDNA PRs compile out with `SGL_IS_RDNA`. Irrelevant for a single GPU.
- **aiter** is not strictly required (`SGLANG_USE_AITER = EnvBool(False)`), but ~42 files import it, 5 unconditionally at module top; RDNA3 users needed a stub package.
- CI: `test/registered/attention/test_deterministic.py` registered for AMD CI (MI300, `stage-b-test-1-gpu-small-amd`), two classes skipped on AMD; `test_amd_deterministic_custom_allreduce.py` on `2-gpu-amd`. No issue mentions deterministic inference on consumer AMD.

### 1.4 RDNA2 limits relevant to SGLang's Triton kernels

- Triton recognises gfx10.3 (`ISAFamily::RDNA2`), `tl.dot` lowers to FMA (section 3) - SGLang's Triton attention/MoE/GEMM kernels compile but run far slower per FLOP than on RDNA3; vLLM issue #38107 reports 1.4-4.2 tok/s on gfx1030 with `TRITON_ATTN` (https://github.com/vllm-project/vllm/issues/38107). For determinism this is benign: FMA accumulation has a fixed, simple order.
- Wave32: SGLang's `num_warps` choices (e.g. `decode_attention.py: num_warps = 1` on HIP) are tuned for wave64 CDNA; occupancy on wave32 is untuned. The HIP AOT `WARP_SIZE=64` host hard-code is the hard break.
- LDS: Triton models RDNA2 with 64 KB (`getSharedMemorySize()` default); gfx942 also has 64 KB, so the gfx942 `topk_dynamic_smem_bytes = 48*1024` budget would carry over; LDS is not the blocker.
- Other Triton GFX10 exclusions: no DPP warp reduce ("GFX10 (RDNA1/2) are excluded", falls back to shuffles), no buffer atomics, no async-copy/pingpong fast paths. None break correctness.
- FP8: `setup_rocm.py` picks FNUZ (gfx942) or E4M3 otherwise; RDNA2 has no FP8 hardware - **unverified** whether SGLang's FP8 paths fail loudly on gfx103x.
- Docs mention `pip install -e "python[all_hip]"` but no HIP extra appears in `python/pyproject.toml` on `main` (**unverified** whether renamed or dropped).

**Assessment:** making SGLang run on the 6700 XT would require patching the arch allowlist and LDS branch, the wave32 `WARP_SIZE` fix (#31803/#32543), `SGL_IS_RDNA` guards, uninstalling the CUDA flashinfer wheel, stubbing aiter imports, `--attention-backend triton --disable-cuda-graph`, and `HSA_OVERRIDE_GFX_VERSION=10.3.0` - and you would be the first to try RDNA2, while RDNA3 users with all of that still hit an unresolved decode hang. The deterministic machinery is architecture-agnostic in principle and worth reading as a reference design (fixed split-KV tiles, 1-stage extend, seeded Gumbel sampling, batch-invariant GEMMs, pinned MoE config).

---

## 2. vLLM on ROCm

Latest release at research time: v0.29.0 (2026-09-09).

### 2.1 Officially supported archs

- Docs (`docs/getting_started/installation/gpu.rocm.inc.md`, https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html): "MI200s (gfx90a), MI300 (gfx942), MI350 (gfx950), Radeon RX 7900 series (gfx1100/1101), Radeon RX 9000 series (gfx1200/1201), Ryzen AI MAX / AI 300 Series (gfx1151/1150)"; "ROCm 6.3 or above".
- `docker/Dockerfile.rocm_base` (https://raw.githubusercontent.com/vllm-project/vllm/main/docker/Dockerfile.rocm_base): `ARG PYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1200;gfx1201;gfx1150;gfx1151`; `ENV AITER_ROCM_ARCH=gfx942;gfx950`; CK flash-attention built with `GPU_ARCHS=$(echo ${PYTORCH_ROCM_ARCH} | sed -e 's/;gfx1[0-9]\{3\}//g')` - all RDNA targets stripped from the CK FA build. Base image `rocm/dev-ubuntu-22.04:7.2.3-complete`.
- `CMakeLists.txt` (https://raw.githubusercontent.com/vllm-project/vllm/main/CMakeLists.txt) is the one place gfx1030 appears: `set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1250;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201")`. Machete/Marlin-style HIP kernels are gfx942/gfx950 only.
- Runtime gating in `vllm/platforms/rocm.py` (https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/platforms/rocm.py):
  ```python
  _ON_GFX1X = any(arch in _GCN_ARCH for arch in ["gfx11", "gfx12"])
  _ON_CDNA  = any(arch in _GCN_ARCH for arch in ["gfx9", "gfx1250"])
  _ON_RDNA  = _ON_GFX1X and not _ON_CDNA
  ```
  gfx1030 matches neither, so RDNA-specific paths (e.g. `use_rocm_custom_paged_attention`'s non-CDNA branch) silently evaluate false; `is_navi()` (`"gfx1" in _GCN_ARCH`) does match.
- Open PR #52391 "[Feature][ROCm] Enable gfx1030 (RDNA2) as a recognized RDNA arch" (2026-08-14, unmerged, https://github.com/vllm-project/vllm/pull/52391): adds gfx1030 to `PYTORCH_ROCM_ARCH`, docs ("Radeon PRO V620 / RX 6800/6900 series (gfx1030)"), `_ON_GFX10X`/`on_gfx10x()` in `rocm.py`, and gates in `triton_w4a16.py` and `flash_linear_attention`. Companion issue #54728 (2026-09-01, https://github.com/vllm-project/vllm/issues/54728). Historical: ROCm/vllm PR #327 (2024-12) added `gfx1030;gfx1100;gfx1101`, hit "Target architectures support different types of FP8 formats!", and was reverted.
- ROCm side: TheRock `SUPPORTED_GPUS.md` lists gfx1030 and gfx1031 as build-passing/sanity-tested on Linux (https://raw.githubusercontent.com/ROCm/TheRock/main/SUPPORTED_GPUS.md); the ROCm 6.4.1 compatibility matrix lists gfx1030 as a compute target (https://rocm.docs.amd.com/en/docs-6.4.1/compatibility/compatibility-matrix.html). The blocker is vLLM's arch lists and kernels, not ROCm.

### 2.2 Field evidence for gfx1030

| Report | Date | Hardware / stack | Outcome |
|---|---|---|---|
| vllm #13648 (https://github.com/vllm-project/vllm/issues/13648) | 2025-02-21 | RX 6700 XT (gfx1030), ROCm 6.3, vLLM 0.7.3, torch 2.6.0+rocm6.2.4, TP=2 | Crash in `rocm_flash_attn.py` (`_sdpa_attention()` arity), fixed by PR #13650; shows V0-era builds ran on a 6700 XT |
| vllm #7512 (https://github.com/vllm-project/vllm/issues/7512) | 2024-08-14 | gfx1031, vLLM 0.5.4 | GPTQ/AWQ quantization check failure |
| vllm #38107 (https://github.com/vllm-project/vllm/issues/38107) | 2026-03-25 | 2x Radeon PRO V620 (gfx1030), ROCm 7.2, torch 2.11+rocm7.2, source build, `TRITON_ATTN`, Qwen3-8B TP=2 | 1.4-4.2 tok/s. AMD's chejh-amd: "RDNA 2 does not have hardware support for bfloat16 ... falls back to software emulation for every matrix multiplication"; workaround `--dtype float16` |
| vllm #45372 (https://github.com/vllm-project/vllm/issues/45372) | 2026-06-12 | RX 6800 (gfx1030), ROCm 7.2, `TORCH_ROCM_ARCH=gfx1030`, fp16 GPTQ Qwen3-4B | v0.21: 30-60 tok/s; v0.22: 1-2 tok/s after the GPTQ consolidation commit; open |
| PR #52391 benchmarks | 2026-08 | 4x V620, ROCm 7.2, torch 2.12 | Qwen2.5-7B-GPTQ: 8-10 tok/s at c=1, 40-52 tok/s at c=8 |
| PR #52919 (docs, self-closed) | 2026-08-19 | RX 6750 XT / RX 6800, Windows, vLLM 0.19.1, ROCm 7.15 (TheRock) | 59-62 tok/s Qwen3.5-4B AWQ; community fork |

No community Docker image for gfx1030 was found: lemonade-sdk/vllm-rocm ships gfx1151/1150/120X/110X only; nlzy/vllm-gfx906 is gfx906-only and archived; `rocm/vllm` Docker Hub tags cover gfx950/gfx94X/gfx120X/gfx115X/gfx110X (**unverified** directly, per search snippet). Whether a gfx1031 card with `HSA_OVERRIDE_GFX_VERSION=10.3.0` works with current V1 vLLM is **unverified** (the only 6700 XT reports predate V1).

### 2.3 `VLLM_BATCH_INVARIANT`

- Module moved to `vllm/model_executor/determinism/batch_invariant.py` (+ `batch_invariant_configs.py`); the old `layers/batch_invariant.py` path is gone and `vllm_is_batch_invariant()` no longer exists - call sites read `envs.VLLM_BATCH_INVARIANT` (https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/determinism/batch_invariant.py). Env definition in `vllm/envs.py`: "Requires NVIDIA GPU with compute capability >= 9.0" (docs say 8.0). Landed in v0.11.1 (2025-11-19). Docs: https://docs.vllm.ai/en/latest/features/batch_invariance/ ("beta"; supported hardware: NVIDIA >= SM80 and Intel XPU with Triton; **no AMD**). Tracker #27433 (https://github.com/vllm-project/vllm/issues/27433) lists "AMD support" as unchecked -> PR #52231.
- `init_batch_invariance()` (called from `vllm/v1/worker/gpu_worker.py`): `resolve_tuned_matmul_configs()`, `override_envs_for_invariance()` (`VLLM_ALLREDUCE_USE_SYMM_MEM=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `NCCL_ALGO=allreduce:tree`, `NCCL_PROTO=Simple`, `NCCL_MIN/MAX_NCHANNELS=1`, `NCCL_LAUNCH_MODE=GROUP`, `VLLM_USE_AOT_COMPILE=0`), `enable_batch_invariant_mode()`, TF32 off (`fp32_precision = "ieee"`).
- `enable_batch_invariant_mode()` platform gating (quoted from `main`):
  ```python
  key = current_platform.dispatch_key
  if current_platform.is_cuda():
      if current_platform.is_device_capability_family(80):
          _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, key)
          ... addmm, matmul, linear ...
      else:
          os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
          os.environ["CUBLASLT_WORKSPACE_SIZE"] = "1"
  elif current_platform.is_xpu():
      ... registers mm/addmm/matmul/linear with tensor-descriptor Triton matmul ...
  ```
  then, for all non-XPU platforms, registers `aten::_log_softmax`, `softmax`, `_softmax`, `mean.dim`, `bmm` overrides and sets reduced-precision reductions off and `preferred_blas_library("cublaslt")`. **There is no `is_rocm()` branch**: on ROCm no GEMM override is installed, and the cuBLAS workspace variables are meaningless, so GEMM batch invariance is left to hipBLAS/hipBLASLt (which, per the local test above, is not batch-invariant). `RMSNorm.forward_cuda` does route to `rms_norm_batch_invariant`; `UnquantizedLinearMethod.apply` -> `linear_batch_invariant` when `is_cuda_alike() or is_xpu()` (ROCm counts as cuda-alike), so linear layers do get the Triton path through that route (behaviour of `resolve_tuned_matmul_configs()` with an AMD device name **unverified**). The matmul tile selection contains a shared-memory check (`get_max_shared_memory_bytes() > 106496` -> N=256 tiles, else N=128/32 with fewer stages) added to avoid Triton out-of-resource errors on small-LDS parts.
- Config effects: `disable_custom_all_reduce = True` (`config/parallel.py`); cascade attention off and allreduce+RMSNorm fusion off (`config/vllm.py`); FA AOT scheduler off. Not disabled: CUDA graphs and torch.compile (supported per #27433), prefix caching (not auto-disabled; listed as "not yet supported"), chunked prefill. MoE: `fused_moe.py` fixed Triton config `{BLOCK_SIZE_M:64, N:64, K:32, GROUP_SIZE_M:8, SPLIT_K:1}`, modular kernels must declare `_supports_batch_invariance()`. Sampling: no sampler code modified; relies on per-request `seed`.
- Attention: base class `supports_batch_invariance() -> False` (`vllm/v1/attention/backend.py`); True for `FLASH_ATTN` (`num_splits=1`), `FLASHINFER` (fixed 2048/4096 split sizes, `disable_split_kv=True`, but test suite disables it pending FlashInfer #2424 where `cta_tile_q` flips 64<->128 with batch composition), `FLEX_ATTENTION` (`block_m=block_n=16`), `TRITON_ATTN`, `TRITON_MLA` (1 split), `FLASH_ATTN_MLA`. FA4 refused. GDN/Mamba raise (#42960). **ROCm backends `ROCM_ATTN`, `ROCM_AITER_FA`, `ROCM_AITER_UNIFIED_ATTN` do not override it** (default False). Test matrix `tests/v1/determinism/utils.py`: cuda `["FLASH_ATTN","TRITON_ATTN","FLEX_ATTENTION"]`, xpu `["TRITON_ATTN"]`.
- Tests: `tests/v1/determinism/test_batch_invariance.py` asserts `torch.equal` on per-step logprobs for bs=1 vs bs=N (T=0.6, seed, logprobs=5) and a needle-in-batch text test up to bs=128; `test_matmul_batch_invariant.py`, `test_rms_norm_batch_invariant.py`, CUTLASS/NVFP4 tests; `test_simple_generation` sets `enable_prefix_caching=False`.
- **PR #52231 "[Feature] Support batch invariance on ROCm"** (mawong-amd, 2026-08-14, open, needs rebase; https://github.com/vllm-project/vllm/pull/52231): tested on gfx942/gfx950 only; keeps a fixed-rank-order custom all-reduce on ROCm, adds `TritonFP8ScaledMMLinearKernel`, ROCm E2E tests use `TRITON_ATTN`, `rdna_hybrid_w4a16.py` returns `can_implement() == False` under batch invariance. Whether it adds an `is_rocm()` GEMM override branch is **unverified** from the file list.

### 2.4 ROCm attention backends usable on gfx1030

From `_get_backend_priorities()` in `vllm/platforms/rocm.py`: `ROCM_ATTN` (always), `ROCM_AITER_FA` / `ROCM_AITER_UNIFIED_ATTN` (need aiter, built only for gfx942/gfx950), `TRITON_ATTN`, `TURBOQUANT`; MLA: `TRITON_MLA` (or aiter variants). On gfx1030: aiter backends unusable; `ROCM_ATTN` degrades to pure Triton (`use_rocm_custom_paged_attention()` is False without `_ON_GFX1X`/CDNA) and is not batch-invariance-capable; **`TRITON_ATTN` is the only backend that is both runnable on gfx1030 and declared batch-invariant** - force with `--attention-backend TRITON_ATTN`.

**Would `VLLM_BATCH_INVARIANT=1` work on gfx1030 today?** Not as a guarantee: GEMMs stay on hipBLAS (no ROCm branch), the softmax/mean/bmm/rms_norm Triton overrides would compile (Triton handles gfx10), `TRITON_ATTN` is required, fp16 only in practice, and the pending AMD PR targets CDNA. Watch #52391, #54728, #52231, #45372.

---

## 3. Triton on gfx1030

### 3.1 Which Triton ships with PyTorch ROCm wheels

`pytorch-triton-rocm` / `triton-rocm` is built from **upstream** triton-lang/triton (`.github/scripts/build_triton_wheel.py` clones `https://github.com/openai/triton`, only the package name changes for ROCm), not the ROCm/triton fork.

| PyTorch branch | Triton | pinned commit (all in triton-lang/triton) |
|---|---|---|
| release/2.7 (rocm6.3 index) | 3.3.1 | `96316ce5` |
| release/2.8 | 3.4.0 | `11ec6354` |
| release/2.9 | 3.5.1 | `bfeb0668` |
| release/2.10 | 3.6.0 (`triton-rocm`) | `8fedd49b` |
| release/2.11 | 3.6.0 | `9844da95` |
| main | 3.8.0 | `c01b6774` |

Sources: `.ci/docker/ci_commit_pins/triton.txt` and `.ci/docker/triton_version.txt` per branch; wheel metadata on https://download.pytorch.org/whl/rocm6.4/torch/ (`torch-2.9.1+rocm6.4` requires `pytorch-triton-rocm==3.5.1`; the rocm6.4 index tops out at torch 2.9.1 / Triton 3.5.1; torch 2.10+ is on rocm7.0 with `triton-rocm`). Official PyTorch ROCm wheels are built with `PYTORCH_ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"` (https://github.com/pytorch/pytorch/blob/release/2.8/.ci/docker/manywheel/build.sh), so torch itself has native gfx1030 code objects.

### 3.2 Official architecture support in the Triton AMD backend

- README lists only "AMD GPUs (ROCm 6.2+)" (https://raw.githubusercontent.com/triton-lang/triton/main/README.md); no gfx list, no release note ever dropped gfx10 (release notes 3.4.0-3.8.0 mention only gfx950/gfx1250/RDNA3/RDNA4 work plus gfx906 support added by PR #9628, 2026-03-06).
- ISA classification: 3.3.x-3.5.x `TargetUtils.cpp`: `GK_GFX1030..GK_GFX1036 -> ISAFamily::RDNA2` (https://raw.githubusercontent.com/triton-lang/triton/release/3.5.x/third_party/amd/lib/TritonAMDGPUToLLVM/TargetUtils.cpp); main `TargetFeatures.cpp`: `if (major == 10 && minor == 3) return ISAFamily::RDNA2;` (https://raw.githubusercontent.com/triton-lang/triton/main/third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp).
- Warp size: `compiler.py`: `warp_size = 32 if gfx_major >= 10 else 64` (PR #3083, 2024-02); LDS: `getSharedMemorySize()` = 64 KB default (160 KB CDNA4, 320 KB GFX1250).
- `tl.dot` lowering: `AccelerateAMDMatmul.cpp`: `getMfmaVersion` and `getWmmaVersion` both return 0 for RDNA2, so only the `AccelerateBlocked` FMA pattern applies (https://raw.githubusercontent.com/triton-lang/triton/main/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp). `DotOpToLLVM/FMA.cpp`: f16xf16->f32 uses `llvm.amdgcn.fdot2` (`v_dot2_f32_f16`, available on gfx10.3 via `FeatureDot10Insts`); bf16 is only treated as a legal v_dot on CDNA4, otherwise upcast to f32 and `llvm.fmuladd.f32`; i8 uses `sdot4`. Note `v_dot2_f32_f16` fuses two products per instruction, so its rounding differs from two sequential FMAs - deterministic, but pick your reference accordingly.
- Other gfx10 gaps: DPP warp reduce excluded ("GFX10 (RDNA1/2) are excluded", PR #5019) -> xor-shuffle tree; no buffer atomics; in-thread transpose, async copy, pingpong all gated to other archs; no `is_hip_rdna2` test predicate; **CI runs on gfx90a/gfx942/gfx950 only** (`integration-tests-amd.yml`).
- Maintainer stance: "Maintaining more backends requires a big effort on our end, and our small team (4 people) can't afford it" (#8992, 2025-12); PR #10795 closed "we have no plans to support these very old architectures" (2026-07). gfx906 nevertheless landed via a contributor PR.
- ROCm docs: 6.4.1 compatibility matrix lists gfx1030 / RDNA2; the 6.4.1 system-requirements page lists gfx1030 only as Radeon PRO W6800 and V620 - RX 6700 XT is not listed (https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.1/reference/system-requirements.html). The Radeon landing page says ROCm 7.2.1 supports RDNA4 and select RDNA3 consumer GPUs; RDNA2 consumer parts are not covered.

### 3.3 Evidence from issues

- triton-lang/triton #3704 "Low performance on RX 6800XT / Navi21" (2024-04, Triton 3.0 dev): matmul tutorial compiled and ran ~6-7x slower than rocBLAS; attention tutorial needed `order=(1,0)`; closed by reporter 2025-08 "Performance is now acceptable" (https://github.com/triton-lang/triton/issues/3704).
- **bf16 regression in Triton 3.6.0-3.8.0**: `ElementwiseOpToLLVM.cpp` lowers any bf16xbf16 elementwise multiply on `isRDNA(isaFamily)` (includes RDNA2) to `llvm.amdgcn.fdot2.bf16.bf16`, which requires `HasDot9Insts` - absent on gfx10.3 -> `LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.fdot2.bf16.bf16` (reproduced in a gfx1030 setup README, workaround `dtype=torch.float16`); on RDNA3/4 it silently truncated instead of rounding (#11217). Fixed on main by PR #11227 (merged 2026-08-14, after the 3.8.0 cut; https://github.com/triton-lang/triton/pull/11227). Triton 3.5.x and earlier use the fp32 upcast path and are unaffected - consistent with the local bf16 success on 3.5.1.
- PR #8633 "[AMD] Fix nondeterministic atomic tests failure on RDNA" (merged 2025-11-11): tensor `atomic_rmw` returned ~1.56% wrong elements on RDNA in Triton <= 3.5.x due to an unguarded LDS store; RDNA2 shares the code path (**unverified** on gfx1030). Avoid atomics anyway.
- No gfx1030-specific reports of illegal instructions or silently wrong non-atomic numerics were found in triton-lang/triton or pytorch/pytorch (searches for gfx1030, gfx1010, 6700 XT, 6900 XT, HSA_OVERRIDE_GFX_VERSION, RDNA2 inductor). Inductor's `get_warp_size()` is arch-correct (32 for gfx10+); Inductor's CK backend is gated to gfx90a/gfx942/gfx950.

### 3.4 Determinism properties of the backend

- Non-atomic reductions: generic `ReduceOpToLLVM.cpp` uses a fixed xor-butterfly ("Changing this breaks a ton of bitwise comparisons so we stick with the legacy inverse order") after `warpReduce` returns false on GFX10; cross-warp stages go through shared memory in fixed order. The tree depends on layout (`num_warps`, block sizes), so batch invariance is the kernel author's job. Because DPP reduce is on for gfx90a+/gfx11+ but off for gfx10, the same `tl.sum` can be bit-different on gfx1030 vs MI300/RDNA3.
- Atomics are the only nondeterminism source (split-K/stream-K GEMMs use `tl.atomic_add`); avoid them.
- Flags affecting bitwise results: `enable_fp_fusion` default True (`TRITON_DEFAULT_FP_FUSION`), `allow_flush_denorm=False` (`denormal-fp-math-f32=ieee`), `__HIP_FTZ=True` forces `exp2` to flush denorms, `__oclc_unsafe_math_opt=False`. Pin `enable_fp_fusion` per kernel if comparing against a non-fused reference.
- Inductor on main has `TORCHINDUCTOR_DETERMINISTIC=1` and `TORCHINDUCTOR_BATCH_INVARIANT=1` ("stable per-sample compiled kernel across batch sizes") - presence in 2.9.x **unverified**.

---

## 4. tinygrad

Latest release 0.14.0 (2026-08-24); master at commit d22b12b4 (2026-09-09). Note that master has already collapsed the optimizer vocabulary (`OptOps.TC/SPLIT/PADTO/SWAP` with `AxisType`), whereas the 0.14.0 wheel tested locally still prints `OptOps.GROUPTOP/UPCAST/LOCAL`.

### 4.1 AMD backend support for gfx1030/RDNA2

- `tinygrad/runtime/ops_amd.py` (`AMDDevice.__init__`, after interface selection, so it applies to KFD, PCI/AM and USB alike):
  ```python
  self.arch = "gfx%d%x%x" % self.target
  assert (self.target in ((9,4,2),(9,5,0))) or self.target[0] in (11, 12), f"Unsupported arch: {self.arch}"
  ```
  (https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/ops_amd.py#L855-L857). Confirmed locally on 0.14.0: `AssertionError: Unsupported arch: gfx1031`; patching the assert only moves the failure to `no such autogen: soc_10` - gfx10 register tables were removed.
- Dropped by PR #16017 "am: only support CDNA3/4 and RDNA3/4" (merged 2026-05-02, https://github.com/tinygrad/tinygrad/pull/16017): removed the gfx10 ctl-stack, `mmCP_COHER_START_DELAY`, wave-count and NBIO 3.3->2.3 override paths and changed `docs/runtime.md` from "RDNA2 or newer GPUs" to "CDNA3, CDNA4, RDNA3 or RDNA4 GPUs". v0.12.0 (2026-01-12) is the last release with gfx10 code paths; v0.13.0 and v0.14.0 carry the assertion.
- RDNA2 history: worked best-effort Mar 2025 - May 2026 (issue #9363 RX 6900 XT fixed by PRs #9365/#9384; #9665/#9700 "Fix RDNA2 support"). Issue #15005 (2026-02): geohot "mostly we are targeting RDNA3/RDNA4/CDNA". **PR #17666 "add RDNA2 full support"** (2026-08-21, +9292/-52, "Tested GPT-2 and Stable Diffusion with RX 6700 XT (gfx1031)") closed unmerged; geohot: "We don't plan to test or support it ... I would spend 10 lines max on RDNA2" (https://github.com/tinygrad/tinygrad/pull/17666). Its runtime hunks (allow 10 in the assert, uncached GTT for cpu_access buffers, 1 KB scratch alignment on gfx9/10, `num_waves` divisor, NBIO override, `COMPUTE_TMPRING_SIZE_GFX10`, `"gfx1030": []` in `tc.get_amd`) are the inventory of what a fork would need.
- AM driver: `docs/developer/am.md`: "userspace driver targeting AMD's RDNA3/RDNA4"; `support/am/amdev.py` firmware loading is for gc>=11 and gfx9.4/9.5. Not for gfx10.
- Compilers: default `HIPRenderer` -> `HIPCompiler` -> comgr (asserts `comgr not available` otherwise; needs ROCm's `libamd_comgr`); `DEV=AMD:LLVM` uses system libLLVM >= 18 (no ROCm needed); `HIPCCRenderer` shells out to `hipcc`; the in-house assembler has tables only for rdna3/rdna4/cdna. `AMD=1`/`AM=1`/`AMD_LLVM=1` are deprecated in favour of `DEV=AMD`, `DEV=AMD:LLVM`, `DEV=PCI+AMD` (asserts in `device.py`).
- Tensor-core trap: `codegen/opt/tc.py: get_amd(arch)` returns `amd_rdna3` (WMMA) for any arch not in `{gfx942, gfx950, gfx1200, gfx1201}` - confirmed locally: `DEV=HIP` fails to compile `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` on gfx1030 until `TC=0`.
- `ops_hip.py` exists but is not in `ALL_DEVICES` and has no CI; `DEV=CL` (OpenCL) works locally. CI AMD coverage: emulated gfx1100, gfx1201, gfx950 only.

### 4.2 How kernel shapes / reduction splitting are decided, and determinism

- Pipeline (master): scheduler emits kernels with `KernelInfo()`; `codegen/opt/postrange.py: apply_opts(ast, ren, beam)` applies, in order of precedence: `ast.arg.opts_to_apply` if set; else BEAM search if `BEAM>=1`; else `hand_coded_optimizations(k)` unless `NOOPT` (https://github.com/tinygrad/tinygrad/blob/master/tinygrad/codegen/opt/postrange.py#L305-L322).
- **(a) Deterministic given the AST?** Yes for hand-coded: a pure function of the kernel's shapes/strides/renderer plus env vars; compiled programs are memoised by `(ast.key, renderer type, target, NOOPT, USE_TC, TC_OPT, ...)`. Caveat: `MV`, `MV_BLOCKSIZE`, `MV_THREADS_PER_ROW`, `MV_ROWS_PER_THREAD`, `OCCUPANCY_FLOOR` are read via `getenv` and are not in that key (`codegen/opt/heuristic.py`).
- **(b) Does the matmul AST depend on batch size?** Yes. `Tensor.dot` is `(x*w).sum(-1)` on a broadcast `(B,N,K)` tensor, so B is a GLOBAL axis of the same reduce kernel and every heuristic that reads `full_shape`/`output_shape` sees it: matvec path (`SPLIT(reduce, (8, GROUP_REDUCE))` when a global axis is divisible by `MV_BLOCKSIZE*MV_ROWS_PER_THREAD`), grouping (`prod(output_shape[upcastable]) <= 2048` -> 16-way GROUP_REDUCE of K), upcast loop (`>= 1024` outputs), unroll of K by 4 conditioned on `upcast_size()`, local sizes over global axes. Confirmed locally (section "Local experiments"): three different kernels for M=1/7/256 and a bitwise difference on row 0.
- **(c) BEAM** (`codegen/opt/search.py`): candidates timed with `_time_program` (min of 3 runs), machine- and noise-dependent; result cached in sqlite `CACHEDB` keyed by `{ast.key, amt, allow_test_size, device type, renderer suffix}` (`CACHELEVEL>=1`, `IGNORE_BEAM_CACHE=1` bypasses). Deterministic only if you ship `cache.db`; still not batch-invariant.
- **(d) Pinning**: `KernelInfo.opts_to_apply` (`uop/ops.py`) makes `apply_opts` apply an exact `Opt` list and skip BEAM/heuristics - but the scheduler always emits `KernelInfo()` and there is **no public Tensor-level API** to set it; you would monkey-patch `apply_opts`. `NOOPT=1` is batch-invariant by construction (one work-item per output, sequential K loop) but global and unusably slow (confirmed locally). `Tensor.custom_kernel(...)` allows hand-written UOp kernels (tinygrad's own `tinygrad/llm/kernels/amd.py` fast paths use it, gated to gfx11 + HIPRenderer). GROUP_REDUCE is order-deterministic and atomic-free (`fix_group_for_reduce`: partials to LDS, then a plain loop); a grep of the renderers finds only memory fences, no atomic RMW reductions. So given a fixed `Opt` list the reduction tree is fixed; the problem is solely that the *choice* varies with B.
- **(e) Scheduler-level split**: `schedule/prepare.py: split_reduceop` (`SPLIT_REDUCEOP=1` default) fires only when >= 32768 elements are reduced per output (`REDUCEOP_SPLIT_THRESHOLD`), then splits with the largest divisor `d` in `[8, min(256, 2^22 / num_outputs)]` - since `num_outputs = B*N`, the split factor depends on B once `B*N > 16384`. Irrelevant for K < 32768 (Llama-8B/70B), relevant for 405B-class FFNs; `SPLIT_REDUCEOP=0` disables.
- TinyJit specialises on exact input shapes; llama's `forward_jit` is used only for `(1,1)` token shapes.

### 4.3 LLM inference examples / serving

- `examples/llama3.py`: Bottle server with `/v1/completions` and `/v1/chat/completions` (streaming only), batch size 1, one request at a time, module-global prefix cache, no queue/lock, no continuous batching, no paged KV. `extra/models/llama.py`: static KV cache sized on the first-seen batch. `examples/gpt2.py`: replicates one prompt `batch_size` times.
- New `tinygrad/llm/` package (`serve.py`, `model.py`, `gguf.py`, `kernels/amd.py`): single KV cache with token-prefix reuse, `prefill_jit`/`rollout_jit`, sequential request handling (open PR #17359 "add threading to the http server" suggests single-threaded today - **unverified** beyond the title). No paged attention or multi-request scheduler anywhere; issue search for "continuous batching" returns nothing.

### 4.4 Assessment

Not a sensible substrate for this machine:

1. Hardware support is the blocker: every current release asserts on gfx10 for every interface, the register tables are gone, and the maintainer has declined RDNA2 in writing. You would run a fork re-verified against a runtime that changes weekly; the last unpatched release (v0.12.0) is three major refactors behind (rangeify, `DEV=`, OptOps collapse). The unofficial `DEV=CL` and `DEV=HIP TC=0` routes work today but are un-CI'd and unlisted.
2. Kernel generation is inspectable and atomic-free, but not batch-invariant by default and the pinning hook is private; achieving "K order depends only on K" means writing your own opt-selection policy for every reduce kernel - comparable to writing the handful of Triton kernels directly, where the reduction tree is explicit in source.
3. BEAM is wall-clock based and off the table for reproducibility.
4. RDNA2 has no WMMA, which removes tinygrad's main codegen advantage; what remains is FMA code through comgr/LLVM with less control over the loop nest than a hand-written kernel. The 0.14.0 fp16 matmul also multiplies in half before accumulating (0.05 max error vs 6e-4 for a proper fp32-accumulate GEMM).
5. No serving infrastructure: continuous batching, paged KV and request scheduling would all be greenfield.

Where it would be attractive: on RDNA3/RDNA4/CDNA3, the fully inspectable AST -> `Opt` list -> source pipeline (`DEBUG=3` logs applied opts) with `opts_to_apply` is a strong base for proving batch invariance kernel by kernel. On gfx1030, PyTorch/ROCm 6.4.1 (which officially supports gfx1030) plus a few custom Triton/HIP kernels with tiling fixed per (K, dtype) and independent of M/B is the pragmatic path.

---

## 5. Batch-invariant reference implementations

### 5.1 Thinking Machines `batch_invariant_ops`

- Blog "Defeating Nondeterminism in LLM Inference" (2025-09-10, https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/): nondeterminism comes from lack of batch invariance plus varying server batch sizes, not primarily from atomics. Prescriptions: RMSNorm one row per core; matmul fixed tile config, no split-K (~20% perf cost); attention fix the split **size** not the split count; update KV cache/page table before the attention kernel. Qwen3-235B: 80 unique completions out of 1000 at T=0 without, 1 with.
- Repo https://github.com/thinking-machines-lab/batch_invariant_ops: **MIT license** ("Copyright 2025 Thinking Machines Lab"); `pyproject.toml` deps `torch`, `triton` only; 7 commits, last 2025-11-04 (PR #7 "fix torch.mean"; PR #5 XPU support; PR #9 output type fix).
- Kernels in `batch_invariant_ops/batch_invariant_ops.py`: `matmul_kernel_persistent` (-> `aten::mm`, `aten::addmm`), `_log_softmax_kernel` (-> `aten::_log_softmax`, last dim only), `mean_kernel` (-> `aten::mean.dim`, fp32 output). **No** `bmm`, `linear`, `rms_norm` or attention kernel; `get_batch_invariant_attention_block_size()` returns `block_m=block_n=16` for FlexAttention (the vLLM PoC PR #24583 forced FlexAttention with 16x16 blocks). Tile configs: bf16 128x128x64, fp16 128x256x64, fp32 128x128x32, `num_stages=3`, persistent grid `min(NUM_SMS, tiles)`, fp32 accumulator, no split-K.
- Installation: `torch.library.Library("aten","IMPL")` with `dispatch_key = torch.accelerator.current_accelerator().type.upper()` -> `"CUDA"` on CUDA **and ROCm** builds; `set_batch_invariant_mode()` context manager. `get_compute_units()` uses `torch.cuda.get_device_properties` (works on ROCm); `mean_dim` asserts `input.is_cuda` (true on ROCm). Nothing imports CUDA-only libraries; `tl.dot` lowers to whatever the target has (FMA on gfx1030).
- Open issues: #23 fp32 inputs go through TF32 on NVIDIA (PR #24 proposes `input_precision="ieee"`; on AMD there is no TF32 - local fp32 result 1.06e-3 vs fp64 is consistent with true fp32); #14 `tl.range(flatten=...)` incompatible with released Triton (did not affect 3.5.1 locally); #12 low matmul performance; #1 FlexAttention block divisibility. **No issue or PR mentions ROCm/AMD.** Local result: bf16/fp16 run unmodified and batch-invariant on gfx1030; fp32 needs `num_stages` <= 2 for 64 KB LDS.

### 5.2 vLLM batch-invariant mode - coverage

Covered: `mm/addmm/matmul/linear` via Triton on SM80-family CUDA and XPU (cuBLASLt workspace pinning on SM90+; **nothing on ROCm**), `linear`/LM-head via `linear_batch_invariant` on `is_cuda_alike()`, `_log_softmax/softmax/_softmax/mean.dim/bmm` Triton overrides, `rms_norm` Triton (non-residual path), TF32 off, reduced-precision reductions off, fixed MoE Triton config with `SPLIT_K=1`, modular MoE kernels opt-in, FP8/CUTLASS fixed-config and NVFP4 paths (NVIDIA), AWQ dequant path (#38670), pinned NCCL tree all-reduce with 1 channel and custom all-reduce disabled, cascade attention off. Attention: FLASH_ATTN (`num_splits=1`), TRITON_ATTN, FLEX (16x16), TRITON_MLA (1 split), FLASH_ATTN_MLA; FlashInfer configured but disabled in tests pending flashinfer #2424; FA4 refused. CUDA graphs and torch.compile supported. Gaps: prefix caching (listed not-yet-supported; tests disable it), speculative decoding (PR #52522), GDN/Mamba (#42960), FlashInfer MLA, AMD (#52231), sampler unchanged (per-request seed), VLM issues (#27059). Tests: `tests/v1/determinism/` (bitwise logprob equality bs=1 vs bs=N at T=0.6 with seed; needle-in-batch up to bs=128).

### 5.3 SGLang deterministic mode - coverage

Covered: vendored batch-invariant `mm/addmm/bmm/_log_softmax/mean.dim/rms_norm` (Triton, optional DeepGEMM bf16 on Hopper/Blackwell), unquantized BF16 GEMM forced to cuBLAS instead of CuTe DSL on SM10x; attention FlashInfer (fixed 2048/4096 split tiles, ragged prefill disabled, no radix cache), FA3/FA4 (`num_splits=1`), Triton (256-token decode split tile, unified 1-stage extend, radix cache kept with 4096-aligned chunk truncation and finished-request insertion disabled); CUDA graphs supported; seeded Gumbel-max sampling with murmur hash in fp64; MoE fixed Triton config, `cutedsl` GEMM forbidden; TP: pinned NCCL channels on CUDA, deterministic 1-stage HIP all-reduce on AMD (PR #15340); DeepSeek-V3, Qwen3-Next, FP8 kernels, DP attention marked done in #10278. Gaps: EP, speculative decoding, FlashInfer + radix cache, DeepEP deterministic sorting, Blackwell TP4 (#11513 open), top-k/top-p renormalisation caveat in seeded sampling, seed forwarding bugs on OpenAI endpoints (#15481/#15746). Cost: 25-45% latency (blog). RL follow-ups (Miles blog 2025-11-19) claim exactly 0 KL train/inference mismatch using FA3, DeepGEMM, TML kernels and torch.compile.

### 5.4 AMD/ROCm mentions

- `batch_invariant_ops`: none (but portable; verified locally on gfx1030).
- vLLM: docs NVIDIA/XPU only; no ROCm branch in `enable_batch_invariant_mode`; PR #52231 (open, CDNA-tested) is the AMD effort; no issue mentions RDNA.
- SGLang: Triton backend tested on MI300 at launch (PR #10694), AMD deterministic all-reduce merged (PR #15340); blog explicitly claims AMD support via Triton; nothing on consumer AMD.
- No primary source for any of the three mentions RDNA2/gfx1030.

---

## Appendix: reproduction snippets

Triton batch-invariance probe (run from the project `.venv`; `@triton.jit` functions must live in a `.py` file, not stdin):

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m*BM + tl.arange(0,BM); rn = pid_n*BN + tl.arange(0,BN); rk = tl.arange(0,BK)
    acc = tl.zeros((BM,BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + rm[:,None]*sam + (k+rk)[None,:]*sak, mask=(rm[:,None]<M)&((k+rk)[None,:]<K), other=0.)
        b = tl.load(B + (k+rk)[:,None]*sbk + rn[None,:]*sbn, mask=((k+rk)[:,None]<K)&(rn[None,:]<N), other=0.)
        acc += tl.dot(a, b)
    tl.store(C + rm[:,None]*scm + rn[None,:]*scn, acc, mask=(rm[:,None]<M)&(rn[None,:]<N))
# compare row 0 of tmm(a[:1], b) against tmm(a, b) with torch.equal; compare torch.mm the same way.
```

TML kernels on gfx1030: `git clone https://github.com/thinking-machines-lab/batch_invariant_ops`, add the clone to `sys.path`, wrap `a @ b` in `set_batch_invariant_mode(True)`; for fp32 patch `"num_stages": 3` -> `1` in `batch_invariant_ops/batch_invariant_ops.py`. Compute fp64 references outside the context manager.

tinygrad: `uv venv tg && uv pip install -p tg/bin/python tinygrad numpy`; `DEV=CL python smoke.py` works; `DEV=HIP ROCM_PATH=/opt/rocm-6.4.1 TC=0 python smoke.py` works; `DEV=AMD` asserts. `DEBUG=2` prints kernel names, `DEBUG=4` prints applied `Opt`s and kernel source; `NOOPT=1` restores batch invariance.
