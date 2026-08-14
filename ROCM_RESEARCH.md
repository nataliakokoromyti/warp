# HIP/ROCm Backend for NVIDIA Warp — Research Report

*Researched 2026-08-10. Repo: fork of [nvidia/warp](https://github.com/nvidia/warp) at
`532b11da3` (upstream/main), branch `rocm-backend`.*

## TL;DR

**The port already exists — do not start from scratch.** AMD is actively porting Warp to
ROCm at [AMD-Ecosystem/warp](https://github.com/AMD-Ecosystem/warp) (formerly `ROCm/warp`;
default branch `amd-integration`, created Dec 2025, last pushed 2026-08-10). It builds and
runs Warp — examples, BVH/mesh queries, tile primitives with a rocWMMA MFMA fast path — on
MI325X (gfx942) with ROCm 7.1/7.2. AMD engineers are simultaneously upstreaming
portability-neutral fixes into nvidia/warp ([PR #1702](https://github.com/NVIDIA/warp/pull/1702),
being shepherded to merge by an NVIDIA maintainer as of Aug 6, 2026).

The highest-leverage strategy is therefore: **base on `amd-integration`, contribute at its
open edges** (HIP graphs, conditional nodes, tile FFT, wave32/RDNA, non-gfx942 archs,
Windows, PyPI wheels — none exist), and follow AMD's pattern of upstreaming
vendor-neutral fixes to nvidia/warp.

The `amd` remote is already configured in this repo:
`git diff $(git merge-base upstream/main amd/amd-integration) amd/amd-integration` shows the
full port: **180 files, +7,335/−2,196 lines** (of which ~3.7k insertions in `warp/native`).

---

## 1. Why this matters

- Warp is the substrate for a growing simulation/robotics stack —
  [Newton](https://github.com/newton-physics/newton), 
  [mujoco_warp](https://github.com/google-deepmind/mujoco_warp), differentiable rendering,
  USD/Omniverse tooling — and upstream Warp is **CUDA-only by policy** (README: "GPU support
  requires a CUDA-capable NVIDIA GPU"; the contributor guide explicitly lists AMD/Metal
  portability requests as out-of-scope feature requests).
- AMD shipped the MI400 series (MI455X/MI430X) plus robotics dev platforms at Advancing AI
  2026, and treats simulation as a recognized CUDA-moat area: the
  [MOAT program](https://github.com/AMD-Ecosystem/moat) ("Porting popular CUDA projects to
  ROCm/HIP, one repository at a time"), a
  [Genesis-on-ROCm fork](https://github.com/AMD-Ecosystem/Genesis), and
  [rocm-simulation](https://github.com/ROCm/rocm-simulation) (Taichi + gsplat) all appeared
  within the last ~8 months.
- Measured payoff exists: on MI325X/ROCm 7.2, mujoco_warp (Unitree G1, 256 worlds) went from
  2.4 ms/step eager to 1.4 ms/step with fixed HIP graph capture
  ([mujoco_warp PR #1556](https://github.com/google-deepmind/mujoco_warp/pull/1556)).

## 2. How Warp works (what a backend must provide)

Pipeline (all real code lives in `warp/_src/`; `warp/*.py` are re-export shims):

1. **Codegen** — `warp/_src/codegen.py` (~7.5k lines) lowers Python AST to **CUDA C++**
   (GPU) or plain C++ (CPU). The generated text is ~95% valid HIP already
   (`blockIdx`/`threadIdx`/`__global__`/`extern "C"` all map 1:1).
2. **Compile** — `Module._compile` (`context.py`) → `wp_cuda_compile_program`
   (`warp/native/warp.cu:4622`) drives **NVRTC in-process** → PTX or CUBIN; MathDx tile ops
   add LTO-IR linked with nvJitLink. CPU path uses an embedded Clang (`warp-clang` lib).
3. **Load/launch** — CUDA **driver API**, dlopen'd at runtime from `libcuda.so`
   (`cuda_util.cpp:189-311` resolves ~120 `cuXxx` entry points via `cuGetProcAddress`);
   `cuModuleLoadDataEx` → `cuModuleGetFunction` → `cuLaunchKernel`.
4. **Runtime services** — streams, events, CUDA graphs (incl. conditional nodes), mempools /
   `cudaMallocAsync`, peer access, IPC, textures/mipmaps, OpenGL interop, timing.

Key structural facts:

- The whole GPU surface is a **stable C ABI of ~120 `wp_cuda_*` functions with `void*`
  opaque handles** (`warp.h` exports 296 `WP_API` functions; 120 are `wp_cuda_*`).
  `warp.cpp:1027-1320` already contains a complete no-op second implementation of that ABI
  for CPU-only builds — proof the ABI is swappable.
- Device kernels for builtins live in 13 `.cu` files (~14.7k lines: sort, scan, reduce,
  BVH, hashgrid, volume, sparse, …), almost all via **CUB/Thrust** → near-drop-in
  **hipCUB/rocThrust**.
- ~45k lines of shared math headers (`builtin.h`, `tile*.h`, `mat/vec/quat/...`) compile
  for host+device via a `CUDA_CALLABLE` macro and are mostly portable. Hazards: hard-coded
  `WP_TILE_WARP_SIZE 32` with ~45 `__shfl_*_sync` sites (AMD CDNA wavefront = **64**), and
  5 inline-PTX sites (3 half/double atomics in `builtin.h`, 1 `isspacep.global` in
  `deterministic.h`).
- Python side: `Device` is a hard two-way cpu/cuda branch (`context.py:4956`);
  `is_cpu`/`is_cuda` are consulted at hundreds of sites; allocators are already a pluggable
  Protocol.
- Upstream has **zero** multi-vendor scaffolding — no Backend abstraction, no HIP mentions
  outside vendored third-party code (nanovdb and cuBQL are, usefully, already HIP-aware).

## 3. The existing AMD port (AMD-Ecosystem/warp)

**Approach: compile-time masquerade, PyTorch-style, but hand-rolled instead of hipify.**
A single new header `warp/native/hip_util.h` (578 lines) `#define`s the CUDA driver +
NVRTC symbol surface onto HIP (`nvrtcCompileProgram→hiprtcCompileProgram`,
`nvrtcGetCUBIN→hiprtcGetBitcode`, `cudaX→hipX`, `CUDA_VERSION→HIP_VERSION`) so the
existing `cuda_util.cpp`/`warp.cu` backend compiles unchanged under
`__HIP_PLATFORM_AMD__`. Devices still present as `"cuda:0"`; Python adds `Device.is_hip` /
`runtime.is_hip` flags, and arch handling carries a `gfx942`-style suffix instead of
`sm_NN`. They did **not** use hipify — the compat header was smaller than a hipify
pipeline.

Status (verified against the fetched `amd/amd-integration` branch and its PR history):

| Area | Status in AMD port |
|---|---|
| Core compile/load/launch via hipRTC + hipModule APIs | ✅ Working (gfx942, ROCm 7.1.1/7.2.1, TheRock) |
| Version | `1.13.0+rocm.0`, synced to upstream v1.13.0 (now ~430 commits behind upstream main) |
| BVH / mesh queries | ✅ + two perf PRs (#2, #8) + cuBQL integration (#14) |
| Tile matmul | ✅ **rocWMMA MFMA fast path** ("cuBLASDx equivalent", PR #12) |
| Tile Cholesky | 🚧 in progress (branch `amd/rocwmma-tile-matmul-cholesky`) |
| Tile cross-thread ops (wave64) | ✅ HIP guards through `tile.h`/`tile_reduce.h` |
| HIP graph capture | 🚧 PR #15 open; mempool-during-capture merged (#16); multi-stream capture fix lives in [hipgraph-ms](https://github.com/zhihuidu-amd/hipgraph-ms) |
| Conditional graph nodes (`wp.capture_if/while`) | ❌ raises "not supported on HIP/ROCm" |
| Tile FFT (cuFFTDx) | ❌ no AMD device-side FFT exists |
| RDNA / wave32, Windows, archs beyond gfx942 | ❌ Instinct-only for now |
| OpenGL interop examples | ❌ |
| PyPI wheels | ❌ none (`warp-lang-rocm` / `rocm-warp` don't exist) — source build only |
| hipRTC header quirk | Worked around: ROCm 7 comgr doesn't always find clang/HIP headers; `build.py` adds explicit `-I` dirs |

Upstream engagement: [nvidia/warp PR #1702](https://github.com/NVIDIA/warp/pull/1702)
(zhihuidu-amd) fixes three real HIP semantic differences — `hipMalloc(0)` returns nullptr,
`hipMemsetAsync(size=0)` errors, mempool auto-enable conflicts with PyTorch-ROCm's
allocator — and NVIDIA maintainer `shi-eric` is actively reviewing it toward merge. NVIDIA
has made no roadmap statement about AMD support, but demonstrably **accepts
portability-neutral fixes**.

## 4. HIP compatibility assessment (ROCm 6.x/7.x, as of Aug 2026)

| CUDA feature Warp uses | HIP status |
|---|---|
| NVRTC | **hipRTC**: near-parity API; emits arch-specific code objects (no PTX-like portable IR). Header discovery is the sore spot (see workaround above). |
| Driver API module/launch | `hipModuleLoadData/GetFunction/LaunchKernel` — solid 1:1 (the fact that a 578-line `#define` header covers the whole surface is itself the evidence). |
| Streams/events, peer access, IPC | Full parity. |
| Graphs | API exists (`hipGraph*`, `hipGraphExecUpdate`) but behavioral gaps in practice: **no conditional nodes**, stream-capture+mempool bugs (hence PRs #1702/#15/#16 and hipgraph-ms). |
| `cudaMallocAsync` / mempools | Supported; edge semantics differ (zero-size alloc/memset behaviors). |
| `__shfl*` / ballot / warpSize | **The real porting tax**: wave64 on CDNA vs wave32 on RDNA; `warpSize` not compile-time constant; 64-bit ballot masks; `*_sync` variants landed late in HIP. |
| Thread block clusters (`__cluster_dims__`, sm_90+) | No AMD equivalent — must error cleanly. |
| CUB/Thrust | hipCUB/rocPRIM/rocThrust: near-drop-in. |
| **MathDx (cuBLASDx/cuFFTDx/cuSOLVERDx)** | **No AMD equivalent exists.** Substitutes: rocWMMA for device-side MMA (adopted by the port), composable_kernel for fused GEMM, nothing for device-side FFT. Warp ships scalar fallbacks for every tile op, so `WP_ENABLE_MATHDX=0` is always a correct baseline. |
| nvPTXCompiler / nvJitLink | Unneeded on HIP (no PTX stage; skip LTO machinery). |
| hipify-perl/-clang | Mature for API renames but doesn't fix warpSize assumptions or inline PTX; the AMD port skipped it entirely. |

## 5. Precedent projects — lessons

- **PyTorch** (hipify at build + `torch.cuda` masquerade): most successful port ever;
  maximum ecosystem reuse; cost is permanent mapping-table maintenance. The AMD Warp port
  copies this shape.
- **CuPy** (parallel experimental ROCm build): cautionary tale — perpetually experimental,
  broke on ROCm 6 API removals, hard-coded warpSize=64 bugs. Lesson: without CI on real AMD
  hardware, a parallel build rots. (The AMD fork ships a ROCm CI Docker for this reason.)
- **Taichi** (true separate LLVM AMDGPU backend): clean but took years. Warp has a
  precedent seam here — `warp-clang` already JITs via LLVM NVPTX, and an `amdgcn-amd-amdhsa`
  target + ROCm device-libs would be the same shape — but it's not the fast path.
- **Triton / JAX** (vendor-staffed in-tree or plugin backends): gold standard when AMD
  commits headcount — which, for Warp, it visibly has.
- **numba-hip**: AMD-maintained mirror of `numba.cuda` on HIP — direct analogue for
  Python-JIT frameworks, same "mirror the CUDA API, not the CUDA device" philosophy.

## 6. Recommended strategy for this fork

1. **Rebase this work on `amd/amd-integration`** (remote already fetched here) rather than
   porting nvidia/warp main independently. Their base is upstream v1.13.0; upstream main
   has since moved ~430 commits — a rebase/sync of the AMD delta onto current upstream is
   itself a useful, well-scoped contribution.
2. **Pick open edges, don't redo done work.** Genuinely open niches, roughly by
   value/effort:
   - HIP graph capture hardening (their PR #15 + hipgraph-ms + upstream #1702 are all in
     flight — coordinate, don't duplicate).
   - Conditional graph nodes on HIP (blocked on ROCm feature; clean erroring + CPU fallback
     meanwhile).
   - Tile FFT for AMD (hand-rolled device radix kernels or host-side rocFFT fallback — the
     single largest unported feature).
   - RDNA/wave32 support (parameterize `WP_TILE_WARP_SIZE`, widen ballot masks — benefits
     upstream code quality too).
   - Archs beyond gfx942 (MI300X/gfx90a, MI350/gfx950), Windows, and **PyPI wheels** (none
     exist — highest-visibility gap for adoption).
3. **Upstream vendor-neutral fixes to nvidia/warp** following zhihuidu-amd's pattern
   (maintainers review and merge them); keep the HIP backend itself in the fork.
4. **Keep the masquerade architecture** (`hip_util.h` compat header + `is_hip` flag): it
   maximizes reuse of upstream churn. Guard against CuPy-style rot with CI on real AMD
   hardware (the fork's `docker/rocm_ci` exists for this).

## 7. Repo state

- `origin` → `nataliakokoromyti/warp` (fork, synced with upstream)
- `upstream` → `nvidia/warp` (fetched, base of this branch at `532b11da3`)
- `amd` → `AMD-Ecosystem/warp` (fetched; `amd/amd-integration` is the port)
- Branch `rocm-backend` pushed to origin.

Useful commands:

```bash
# Full AMD port diff vs its upstream base (v1.13.0)
git diff $(git merge-base upstream/main amd/amd-integration) amd/amd-integration

# The compat header that does most of the work
git show amd/amd-integration:warp/native/hip_util.h

# What upstream has done since the AMD port's base
git log --oneline amd/amd-integration..upstream/main
```

## Key sources

[AMD-Ecosystem/warp](https://github.com/AMD-Ecosystem/warp) ·
[nvidia/warp PR #1702](https://github.com/NVIDIA/warp/pull/1702) ·
[mujoco_warp PR #1556](https://github.com/google-deepmind/mujoco_warp/pull/1556) ·
[hipgraph-ms](https://github.com/zhihuidu-amd/hipgraph-ms) ·
[MOAT](https://github.com/AMD-Ecosystem/moat) ·
[rocm-simulation](https://github.com/ROCm/rocm-simulation) ·
[HIP RTC docs](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html) ·
[HIP graphs](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html) ·
[HIP 7.0 transition](https://rocm.blogs.amd.com/ecosystems-and-partners/transition-to-hip-7.0-blog/README.html) ·
[HIPIFY](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/) ·
[NVIDIA MathDx](https://docs.nvidia.com/cuda/mathdx) ·
[rocWMMA](https://github.com/ROCm/rocWMMA) ·
[hipify_torch](https://github.com/ROCm/hipify_torch) ·
[numba-hip](https://github.com/ROCm/numba-hip) ·
[Taichi AMDGPU backend](https://github.com/taichi-dev/taichi/issues/412) ·
[CuPy ROCm deprecation](https://github.com/cupy/cupy/issues/8586)

---

## Addendum: MI350X validation campaign (2026-08-11/13)

Executed on a Stanford cluster MI350X (gfx950, ROCm 7.2.0) — first public validation of the
AMD port beyond MI325X/docker. Results:

- **Warp full test suite: 5,590 tests, OK** (79 skips).
- **Current mujoco_warp (main, warp-lang>=1.15 pin): 1,233 passed / 0 failed / 30 skips**
  (skips: no texture hardware on CDNA, no conditional graph nodes in ROCm, torch absent).
- **All 15 benchmarks run, every world converging.** Highlights: franka 2.50M steps/s @ 32,768
  worlds; humanoid 614k; unitree_g1_flat 450k; unitree_g1_hfield_render 88.7k @ 8,192 worlds.
  Cloth needs a larger contact budget than the NVIDIA-tuned config (physics verified correct
  vs CPU reference over 1,000-step rollouts).

Port fixes on this branch (upstream candidates for AMD-Ecosystem/warp): crt.h isfinite/isnan/
isinf undef for hipRTC; grid_stride kwarg (Warp 1.15 compat); Device.is_texture_supported;
zero-size memset/alloc guards (NVIDIA/warp PR #1702 parity); rocWMMA block_dim!=64 scalar
fallback in tile_matmul. mujoco_warp fixes (patches/mujoco_warp-rocm-compat.patch, upstream
candidates for google-deepmind): texture-less rendering, conditional-graph capability gating,
HIP-aware toolkit check, deterministic island slot assignment (fixes latent scheduling-order
nondeterminism present on NVIDIA as well).
