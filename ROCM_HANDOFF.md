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
static_assert; mujoco_warp launches solver tile kernels with block_dim=128).

mujoco_warp (`patches/mujoco_warp-rocm-compat.patch`): texture-less rendering auto-disable;
`graph_conditional` gated on `wp.is_conditional_graph_supported()`; HIP-aware toolkit check;
**deterministic island slot assignment** (replaces scheduling-order-dependent atomic ranks —
latent nondeterminism on NVIDIA too; parity test now passes).

## THE open problem: graph replay performance

Head-to-head vs AMD's published MI325X result (their mujoco_warp PR #1556: G1, 256 worlds,
eager 2.4 ms → graph 1.4 ms/step):

| Mode (G1, 256 worlds, MI350X) | ms/step |
|---|---|
| eager | **4.8** |
| `wp.ScopedCapture` graph | 10.3 |
| GlobalMode multi-stream graph (hipgraph-ms) | 9.2 |

Graphs are a **net 2× regression** here. Diagnosed so far: the step graph has only **356
nodes**, so replay costs ~26 µs/node, vs 1.7 µs/node for a synthetic 1000-node graph
(measured: `rocm-tools/graph_overhead.py`) — ROCm graph replay pipelines real kernels poorly
where eager stream submission overlaps them. Ruled out: solver iteration unrolling (capping
iterations changes nothing), replay input syncs, multi-stream capture alone
([zhihuidu-amd/hipgraph-ms](https://github.com/zhihuidu-amd/hipgraph-ms) gains only 10%).

Next levers, in order:
1. Newer ROCm (7.3+/TheRock nightly) — graph replay pipelining is actively worked on.
2. Port the *unmerged* extras from
   [mujoco_warp PR #1556](https://github.com/google-deepmind/mujoco_warp/pull/1556)
   (pre-allocated solver ctx / tendon scratch, event handling) — AMD's 1.4 ms used these.
3. Profile one graph replay with rocprofv3 to see per-node gaps.
4. Kernel-level optimization for gfx950 (block sizes, occupancy) — helps eager AND graph.
5. Wave64 tile op tuning; rocWMMA coverage beyond 16x16 f32 (tile Cholesky in progress on
   AMD's `amd/rocwmma-tile-matmul-cholesky` branch).

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
