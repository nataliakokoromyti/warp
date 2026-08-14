# Warp + mujoco_warp on MI350X — collaborator handoff

*Goal: make Warp (and by extension mujoco_warp) fully validated and **super optimized** on
AMD Instinct MI350X (gfx950). Status as of 2026-08-13.*

## What this branch is

`rocm-backend` = [AMD-Ecosystem/warp](https://github.com/AMD-Ecosystem/warp) `amd-integration`
(AMD's official ROCm port of NVIDIA Warp, v1.13.0 base) + their HIP graph-capture branch
(their PR #15) + our fixes. Full background: `ROCM_RESEARCH.md`.

## Current state — what's proven

- **Warp full test suite: 5,590 tests green** on MI350X / ROCm 7.2.0 (bare-metal, not AMD's docker).
- **Current google-deepmind/mujoco_warp main: 1,233 passed / 0 failed / 30 principled skips**
  (needs `patches/mujoco_warp-rocm-compat.patch` applied to a mujoco_warp clone — 7 files).
- **All 15 mujoco_warp benchmarks converge.** Highlights (8,192 worlds unless noted):
  franka 2.50M steps/s (32k worlds), humanoid 614k, unitree_g1_flat 450k, hfield_render 88.7k.

## Our fixes (each upstreamable; see git log)

Warp (this repo): `crt.h` isfinite/isnan/isinf undef for hipRTC; `grid_stride` kwarg
(Warp≥1.15 compat); `Device.is_texture_supported` (CDNA has **no texture hardware** —
`hipDeviceAttributeImageSupport=0`); zero-size memset/alloc guards (NVIDIA/warp PR #1702
parity); rocWMMA `block_dim != 64` scalar fallback in `tile.h` (was a compile-breaking
static_assert; mujoco_warp launches solver tile kernels with block_dim=128);
**kernel-only graph capture** (`warp.cu`, HIP-only: while a stream is capturing,
memset/memtile/d2d-memcpy are routed through trivial kernels instead of
hipMemsetAsync/hipMemcpyAsync, so captured graphs contain no blit nodes).

mujoco_warp (`patches/mujoco_warp-rocm-compat.patch`): texture-less rendering auto-disable;
`graph_conditional` gated on `wp.is_conditional_graph_supported()`; HIP-aware toolkit check;
**deterministic island slot assignment** (replaces scheduling-order-dependent atomic ranks —
latent nondeterminism on NVIDIA too; parity test now passes); **per-step scratch cache**
(`scratch_empty/zeros/full` in `warp_util.py` + cached solver/collision contexts — all 46
per-step temporary allocations reuse Data-cached buffers, so warm-captured graphs contain
no mempool memAlloc/memFree nodes).

## Graph replay performance — SOLVED (2026-08-13)

Graph replay was a net 2× regression (eager 4.8 ms vs graph 9–10 ms/step, G1@256; AMD's
published MI325X result in mujoco_warp PR #1556 was eager 2.4 → graph 1.4). Root cause:
ROCm replays kernel nodes fast (~1.7 µs/node) but stalls at fill/copy blit nodes and
mempool alloc/free nodes. The fix has two parts:

1. **Warp: kernel-only capture** (`warp/native/warp.cu`). During HIP stream capture,
   `wp_memset_device`, `wp_memtile_device`, and `wp_memcpy_d2d` launch trivial fill/copy
   kernels instead of hipMemsetAsync/hipMemcpyAsync. Eager execution and CUDA builds are
   untouched. This alone: graph 10.3 → 8.9 ms/step.
2. **mujoco_warp: per-step scratch cache** (in the compat patch, ported from the unmerged
   parts of AMD's PR #1556 onto current main). Every per-step temporary (solver context,
   collision context, sensor/tendon/transmission/implicit-integrator scratch, `nsolving`)
   is cached on `Data` and re-zeroed/re-filled on reuse instead of reallocated, so no
   memAlloc/memFree nodes are captured.

Result (G1, 256 worlds, MI350X, ROCm 7.2):

| Mode | ms/step |
|---|---|
| eager | 4.16 |
| **warm-capture graph** | **2.94** |
| cold-capture graph | 8.72 |

The warm-captured step graph is 288 nodes, **100% kernel nodes** (verified with
`rocm-tools/graph_census.py`; before the fix it carried 34 memAlloc + 34 memFree + blit
nodes). **Caveat: run a few eager steps before capturing** — a cold capture fires the
scratch allocations *inside* the capture and re-inherits the alloc nodes (hence 8.72).
`rocm-tools/alloc_trace.py` verifies zero allocations fire during a warm capture.
Validated: full mujoco_warp suite green with both fixes (1,233 passed / 0 failed).

Remaining perf levers: kernel-level gfx950 tuning (block sizes, occupancy — helps eager
AND graph); wave64 tile op tuning; rocWMMA coverage beyond 16x16 f32 (tile Cholesky in
progress on AMD's `amd/rocwmma-tile-matmul-cholesky` branch). The scratch cache is a
strong upstream candidate for google-deepmind/mujoco_warp (CUDA graphs also carry alloc
nodes); the kernel-only capture is AMD-specific (AMD-Ecosystem/warp).

Also open: cloth benchmarks need `nconmax≈26000` vs the NVIDIA-tuned 2,200 (physics verified
correct vs CPU over 1,000-step rollouts — accounting difference unexplained); event timing
inside captured graphs is unreliable on HIP (breakdown timings read 0/invalid-handle).

## Working on the Stanford cluster

- Node `matx-amd-1` (8× MI350X, ROCm 7.2.0), sbatch only:
  `--account=matx --partition=matx --nodelist=matx-amd-1 --gres=gpu:mi350x:1`, plus explicit
  `--cpus-per-task`/`--mem` (defaults are 1 CPU / 3 GB).
- Use `/matx/u/$USER` for EVERYTHING (repos, venv, caches — set `XDG_CACHE_HOME`,
  `PIP_CACHE_DIR`, `TMPDIR`); home dirs are tiny and often full.
- Toolchain quirk: node has gcc-14 without libstdc++-14-dev → hipcc/hipRTC pick a GCC dir
  with no C++ headers. The job templates in `rocm-tools/slurm/` handle it
  (`HIPCC_COMPILE_FLAGS_APPEND=--gcc-install-dir=.../13`). An admin install of
  `libstdc++-14-dev` would obsolete this.

## Quickstart

```bash
# on the cluster, in /matx/u/$USER
git clone -b rocm-backend https://github.com/nataliakokoromyti/warp.git warp-rocm
git clone https://github.com/google-deepmind/mujoco_warp.git
cd mujoco_warp && git apply ../warp-rocm/patches/mujoco_warp-rocm-compat.patch && cd ..
# edit paths in warp-rocm/rocm-tools/slurm/*.sbatch, then:
sbatch warp-rocm/rocm-tools/slurm/warp_build.sbatch   # build + SAXPY/tile smoke test
sbatch warp-rocm/rocm-tools/slurm/mjw_probe.sbatch    # full mujoco_warp test suite
sbatch warp-rocm/rocm-tools/slurm/mjw_bench.sbatch    # benchmark suite
```

`rocm-tools/` also has the diagnostics used in this effort: `graph_overhead.py` (graph replay
cost vs node count), `g1_msgraph.py` (single- vs multi-stream capture on G1@256; needs
`hipgraph_ms.py` fetched from zhihuidu-amd/hipgraph-ms), `sort_check.py` (segmented sort
correctness), `flex_check.py` (cloth physics vs CPU reference).

## Upstream relationships

AMD actively develops the port (AMD-Ecosystem/warp) and reviews outside fixes; NVIDIA merges
portability-neutral fixes (see nvidia/warp PR #1702). Our five Warp fixes and four mujoco_warp
fixes are all candidates — upstreaming them early keeps this branch small and rebaseable.
The AMD port (and this branch) is ~430 commits behind nvidia/warp main; syncing that forward
is a valuable, separable workstream.

## Graph-replay investigation results (5-agent campaign, 2026-08-13) — historical

*Superseded by the fix above (lever 1, warp-side kernel-only capture + alloc hoisting,
resolved it). Kept for the record of what was measured and eliminated.*

Diagnosis (rocprof, G1@256): graph replay dispatches identical kernels at identical speed as
eager; the entire ~5 ms gap is IDLE inside hipGraphLaunch replay (~62%), concentrated at
recurring positions — before captured fill/memset nodes and around large-dim/tiled kernels
(which replay ~4x slower than eager). The graph carries mempool alloc/free nodes (proven via
the one-exec-per-alloc-graph 801 restriction).

Measured and ELIMINATED: newer ROCm user-space runtime (TheRock 7.14 nightly — zero delta);
stream parallelism (branches verified present in graph — zero delta); node-count reduction
(29% fewer nodes → 9% faster); env/launch knobs (GPU_MAX_HW_QUEUES etc. — noise); warm-capture
pre-allocation (zero delta). Config-only best: euler integrator 8.3 ms graph; CG solver eager
4.4 ms. Warp already instantiates with AutoFreeOnLaunch + hipGraphUpload.

Levers that were listed here: (1) warp-side kernel-only capture — **done, this was the
fix**; (2) split-graph sweep at /matx/u/knatalia/graphtune_agent/graphtune.py — moot;
(3) minimal repro for AMD — still worth filing (kernel-only graphs replay at 1.7 us/node
while blit/alloc nodes cost ~26 us/node; the workaround shouldn't be necessary). Full agent
logs: /matx/u/knatalia/warp-rocm-logs/{prof_g1_16767817,node-sweep-16768261,gtune*-*,
g1-streams-*,runtime-ab-*,isolate-16769579}.out.
