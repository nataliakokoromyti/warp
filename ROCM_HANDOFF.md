# Warp + mujoco_warp on MI350X — collaborator handoff

*Goal: make Warp (and by extension mujoco_warp) fully validated and **super optimized** on
AMD Instinct MI350X (gfx950). Status as of 2026-08-14.*

## What this branch is

**`rocm-117` (current)** = [AMD-Ecosystem/warp](https://github.com/AMD-Ecosystem/warp)
`amd-integeration-dev` (AMD's re-port of Warp onto near-current upstream — **Warp
1.17.0.dev2**, ~17 commits behind nvidia/warp main as of 2026-08-13) + our fixes ported
forward. This natively satisfies mujoco_warp's `warp-lang>=1.15` requirement — no shims.

`rocm-backend` (legacy, kept intact) = AMD's older `amd-integration` (v1.13.0 base) + their
HIP graph-capture PR #15 + our fixes; fully validated but frozen. Full background:
`ROCM_RESEARCH.md`.

## Current state — what's proven (rocm-117, MI350X / ROCm 7.2.0, bare-metal)

- **Warp full test suite: 8,294 tests, 0 failures / 0 errors** (240 principled skips).
- **google-deepmind/mujoco_warp main: 1,233 passed / 0 failed / 30 skips** — now
  **including all render tests**: CDNA software-sampled texture fallback works, where the
  1.13 branch disabled rendering entirely. (Needs `patches/mujoco_warp-rocm-compat.patch`.)
- **G1@256 graph replay: 1.63 ms/step warm-capture** vs 5.2 eager — graphs are a 3.2×
  speedup (on the 1.13 branch: 2.94 graph / 4.2 eager). Captured step graph remains 288
  nodes, 100% kernel nodes. **Eager regressed** vs 1.13 (4.2 → 5.2 ms) — likely the new
  base's scalar-only tile path (rocWMMA was dropped in AMD's re-port); see perf levers.
- Benchmarks beyond G1 not yet re-run on 1.17 (`mjw_bench.sbatch`); 1.13 numbers for
  reference: franka 2.50M steps/s (32k worlds), humanoid 614k, unitree_g1_flat 450k.

## Our fixes (each upstreamable; see git log)

Warp, on the 1.17 base (`rocm-117`): re-enabled **HIP graph capture** (the dev base gated
it off; flipped `supports_graph_capture`, mempool-on-by-default, capture-time allocation);
**kernel-only graph capture** (`warp.cu`: capture-time memset/memtile/d2d-memcpy become
kernels, so graphs carry no blit nodes); **fail non-pooled allocs during any capture on
HIP** (hipMalloc otherwise *invalidates* the capture, and ROCm 7.2 cannot terminate an
invalidated capture — hipStreamEndCapture returns 908 and the stream is stuck forever);
**capture-safe LBVH rebuild** (AMD's HIP builder read tree depth back to the host and
context-synced mid-build; under capture we bottom-up refit with a new
`refit_internal_nodes_kernel` instead); **13 missing `CUDA_CALLABLE` annotations** in
`tile_solve.h`/`tile_cholesky.h`/`tile_radix_sort.h` (hipRTC does not default functions to
device space like NVRTC) and **18 in `texture.h`** (the `cpu_*` sampling fallback — this
is what makes CDNA software-sampled rendering work); `_cluster_dim_target_status` crashed
comparing HIP's gfx arch string with an int; APIC `capture_save` serialized the arch as
int (gfx string now parsed to its numeric id); carried forward from 1.13: `crt.h`
isfinite/isnan/isinf undef for hipRTC, zero-size memset/alloc guards,
`Device.is_texture_supported`, peer-access sticky-error clear. Plus HIP test gating for
NVIDIA-only features (PTX inspection, clusters, IPC event flags, in-graph event timing,
stream-priority timing) and FP-tolerance relaxations in the gfx950 style AMD established.
Obsolete on 1.17: the `grid_stride` shim (native ≥1.15) and the rocWMMA `block_dim != 64`
fallback (the new base has no rocWMMA path at all).

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

| Mode | 1.13 branch | 1.17 branch (current) |
|---|---|---|
| eager | 4.16 | 5.19 |
| **warm-capture graph** | 2.94 | **1.63** |
| cold-capture graph | 8.72 | ~9.5 |

The warm-captured step graph is 288 nodes, **100% kernel nodes** (verified with
`rocm-tools/graph_census.py`; before the fix it carried 34 memAlloc + 34 memFree + blit
nodes). **Caveat: run a few eager steps before capturing** — a cold capture fires the
scratch allocations *inside* the capture and re-inherits the alloc nodes (hence 8.72).
`rocm-tools/alloc_trace.py` verifies zero allocations fire during a warm capture.
Validated: full mujoco_warp suite green with both fixes (1,233 passed / 0 failed).

Remaining perf levers: **restore a matrix-core (rocWMMA/MFMA) tile path on the 1.17 base**
(AMD's re-port is scalar-only, the likely cause of the eager regression 4.2 → 5.2 ms);
kernel-level gfx950 tuning (block sizes, occupancy); wave64 tile op tuning. The scratch
cache is a strong upstream candidate for google-deepmind/mujoco_warp (CUDA graphs also
carry alloc nodes); the kernel-only capture is AMD-specific (AMD-Ecosystem/warp).

## Cross-vendor benchmark comparison (2026-08-14, rocm-117)

MI350X (this port) vs mujoco_warp's published nightly numbers on an **RTX 6000 Ada**
(48 GB workstation card — roughly 1/8th of MI350X's paper specs). Same scenes, same
world counts, near-identical mujoco_warp commits (ours ea8d067, theirs 70c4571):

| scene | worlds | MI350X steps/s | RTX 6000 Ada | NV/AMD |
|---|---|---|---|---|
| unitree_g1_hfield_render | 8192 | 60,132 | 172,200 | 2.9× |
| three_humanoids | 8192 | 359,263 | 1,075,606 | 3.0× |
| mug | 8192 | 150,055 | 467,650 | 3.1× |
| unitree_g1_flat | 8192 | 722,944 | 2,524,705 | 3.5× |
| myoarm | 8192 | 262,368 | 1,392,823 | 5.3× |
| aloha_clutter | 2048 | 59,979 | 359,280 | 6.0× |
| humanoid | 8192 | 687,727 | 5,664,908 | 8.2× |
| franka_emika_panda | 32768 | 2,777,812 | 23,376,118 | 8.4× |
| aloha_sdf | 8192 | 34,987 | 411,178 | 11.8× |
| unitree_g1_hfield | 8192 | 129,060 | 1,678,525 | 13.0× |

Geomean gap **5.6×** against a much weaker card — i.e. the port has large software
headroom. **Measurement caveat**: repeated sweeps on the shared node show ±10–15 %
run-to-run variance on some scenes (g1_flat spanned 602k–723k across three identical-code
runs; hfield is rock-stable at 129k; the G1@256 graph bench is stable at 1.62–1.63 ms).
Single-sweep deltas below that band are not conclusive. The gap decomposes into three
quantified causes:

1. **No conditional graph nodes on HIP** (API absent in ROCm 7.2 *and* clr main): the
   captured solver runs its full fixed budget (10 iterations on G1/humanoid, 5 on franka)
   while NVIDIA's `capture_while` exits at convergence (their measured niter_mean:
   G1 3.0, humanoid 1.4, franka 1.0). With solve at 21–52 % of NVIDIA's step time, this
   alone costs ~2× on G1 and ~3.8× on humanoid. This is the single strongest ask to AMD:
   conditional node support, with these numbers as justification.
2. **Scalar tile math** (no MFMA/rocWMMA) — multiplies the per-iteration solver cost.
3. **Collision kernels untuned for gfx950/wave64** — the two worst scenes (hfield 13×,
   sdf 11.8×) are collision-dominated, pointing at heightfield-CCD and SDF evaluation
   kernels as specific tuning targets.

vs our own 1.13 branch, 1.17 improved: franka +11 %, humanoid +12 %, G1 flat **+61 %**
(450k → 723k). Render scenes are not comparable across branches (1.13 auto-disabled
rendering; 1.17 really renders). Still not running on 1.17: cloth family (known nconmax
overflow, pre-existing) and primitives (rc=1, needs triage); aloha_pot recovered in later
sweeps (~400k steps/s, 6.3× behind NVIDIA). Raw data:
`/matx/u/knatalia/warp-rocm-logs/bench-results-{16809526,16811065,16812544}.log` and the
nightly JSONL files from google-deepmind.github.io/mujoco_warp/nightly.

### The benchmark gap is mostly COLD CAPTURE, not compute (2026-08-17)

Comparing our eager per-step timings (job 16811026) against the same scenes in the
benchmark sweep reveals that `mjwarp-testspeed` captures its graph **cold** — on a fresh
`Data`, before any step has run — so all 46 per-step scratch arrays are allocated *inside*
the graph as mempool nodes and **re-allocated on every replay**. At 8192 worlds those
buffers are hundreds of MB, and ROCm's graph-mode allocation replay is brutally expensive:

| scene (8192 worlds) | our eager | our cold-graph (= benchmark) | NVIDIA (cold-graph) | cold/eager | **eager vs NV** |
|---|---|---|---|---|---|
| unitree_g1_flat | 5.51 ms | 11.33 ms | 3.24 ms | 2.06× | **1.70×** |
| unitree_g1_hfield | 8.05 ms | 63.47 ms | 4.88 ms | 7.88× | **1.65×** |

**Our eager execution is only ~1.7× behind NVIDIA on both scenes** — strikingly consistent,
and a completely different story from the 3.5×/13.0× the sweep reports. Cold capture is
*slower than not using graphs at all* (2× on flat, 7.9× on hfield); the same effect was
already visible at 256 worlds in `g1_warmcap` (cold 8.98 ms vs warm 1.63 ms).

Note NVIDIA's published numbers use the same cold-capturing harness, so the comparison was
methodologically fair — the finding is that **ROCm punishes cold capture far more than CUDA
does**, and that our scratch cache only pays off when the capture is warm. Fix (in the
compat patch): `cli.unroll` now runs 3 warmup steps before capturing, which is also correct
benchmark practice — it moves one-time allocation out of the measured steady state.
Caveat: re-running our sweep warm while comparing against NVIDIA's cold numbers is no
longer strictly apples-to-apples; report both, and treat the ~1.7× eager comparison as the
honest estimate of the compute gap.

Guidance for users unchanged and now doubly important: **warm up before `wp.ScopedCapture`**
(see the warm-capture caveat above). Tools: `rocm-tools/cold_vs_warm.py` (node census +
replay timing for cold vs warm at any scale), `rocm-tools/traj_ab.py` (eager/cold/warm on a
replay trajectory).

### Optimization round findings (2026-08-15)

- **MFMA (rocWMMA) tile matmul restored but restricted to single-wave blocks**
  (`tile_matmul.h`, gated `WP_TILE_BLOCK_DIM == 64`): at mujoco's 16×16 f32 tiles with
  block_dim=128, MFMA-on-wave-0 showed **no benefit** over the cooperative scalar GEMM
  (within node variance) — consistent with upstream's scalar-vs-cuBLASDx crossover note.
  Matrix cores need larger tiles / fused multi-tile kernels to pay off on this workload.
- **CCD kernel register tuning (compat patch)**: the heightfield CCD kernel launched at
  warp's default 256 block with no launch_bounds → hipcc assumes 1024-thread blocks →
  ≤64 VGPRs → spills. With true block size + `_CCD_MIN_BLOCKS=1` on HIP (min-blocks 8
  re-strangles registers: 31 % WORSE), eager hfield collision improved **3.68 → 2.59 ms
  (‑30 %)**. Bench-level hfield is unchanged because graph-mode steps are dominated by
  fixed solver iterations — reinforcing conditional graph nodes as the top lever.
- **Warp bug found**: the kernel cache hash does not cover the `launch_bounds` kernel
  decorator argument — changing it silently reuses the old binary. Worth an upstream fix
  (nvidia/warp); we hash-bust with source comments in the compat patch meanwhile.
- **Intermittent watch**: two different single-test suite failures across runs
  (`test_copy_i2c_...Graph...`, `test_implicit_fields`), both exact-value partial-write
  signatures, each passing in other runs (suite is otherwise 8,294-green). Needs a
  dedicated flake-hunt (run those classes ~50×) before trusting or chasing.

## Known issues on HIP (gated in tests, documented here)

- **Deterministic mode is not ported**: warp 1.17's deterministic subsystem (phase-0
  counting, deterministic scatter/counter compaction) does not hold on ROCm — repeated
  runs reorder atomically-assigned slots. Whole `warp/tests/deterministic/` suite gated
  off HIP. Needs its own porting effort.
- **Event nodes inside graphs are unreliable**: in-graph event timing reads
  invalid handles, and external-event record/wait nodes do not synchronize
  separately-launched graphs (`test_event_external` gated).
- **Conditional graph nodes unsupported** (`is_conditional_graph_supported()` False);
  captured solver loops run fixed iteration counts.
- **Device-side abort loses printf output**: gfx950 HSA queue aborts (intentional traps,
  OOB asserts) fire before device printf flushes; tests accept the HSA error signature.
- Cloth benchmarks need `nconmax≈26000` vs the NVIDIA-tuned 2,200 (physics verified
  correct vs CPU over 1,000-step rollouts — accounting difference unexplained).

## ROCm bugs worth filing with AMD (minimal repros exist in this history)

1. Blit (fill/copy) and mempool alloc/free graph nodes replay at ~26 µs/node vs
   ~1.7 µs/node for kernel nodes (`rocm-tools/graph_overhead.py`) — the reason all the
   capture workarounds above exist.
2. `hipMalloc` during a thread-local stream capture **invalidates** the capture instead
   of failing cleanly like `cudaMalloc`, and an invalidated capture **cannot be
   terminated**: `hipStreamEndCapture` returns 908 and the stream stays in capture state
   permanently (poisons the process). **Root-caused in clr sources**: 7.2's
   `hipStreamEndCapture_common` (hipamd/src/hip_graph.cpp) returns from the invalidated
   branch without resetting the stream's capture status; the erased thread-ownership
   entry then makes retries fail with 908. **Already fixed upstream** in
   ROCm/clr@fa77aed ("clr: Fix stream capture invalidated state reset", 2026-05-30) —
   but only on `develop`, in no release as of 7.2. Ask AMD for a 7.2.x backport instead
   of filing anew. Our warp-side alloc guard stays regardless: it prevents the
   *invalidation* itself, keeping the capture alive rather than merely failing cleanly.
   Lesson: check clr `develop` before filing any of the items below.
3. `hipStreamIsCapturing(NULL)` does not report the calling thread's capture the way
   CUDA's null-stream query does.
4. `hipThreadExchangeStreamCaptureMode(Relaxed)` does not permit side-stream
   `hipMallocAsync` during another stream's capture (returns 900; CUDA allows this).

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
git clone -b rocm-117 https://github.com/nataliakokoromyti/warp.git warp-rocm
git clone https://github.com/google-deepmind/mujoco_warp.git
cd mujoco_warp && git apply ../warp-rocm/patches/mujoco_warp-rocm-compat.patch && cd ..
# edit paths in warp-rocm/rocm-tools/slurm/*.sbatch, then:
sbatch warp-rocm/rocm-tools/slurm/warp_build.sbatch   # build + SAXPY/tile smoke test
sbatch warp-rocm/rocm-tools/slurm/mjw_probe.sbatch    # full mujoco_warp test suite
sbatch warp-rocm/rocm-tools/slurm/mjw_bench.sbatch    # benchmark suite
```

`rocm-tools/` also has the diagnostics used in this effort: `graph_overhead.py` (graph replay
cost vs node count), `graph_census.py` (node-type census of a captured step graph),
`alloc_trace.py` (which allocations fire during capture, with call sites), `g1_msgraph.py`
(single- vs multi-stream capture on G1@256; needs `hipgraph_ms.py` fetched from
zhihuidu-amd/hipgraph-ms), `sort_check.py` (segmented sort correctness), `flex_check.py`
(cloth physics vs CPU reference).

## Upstream relationships

AMD actively develops the port (AMD-Ecosystem/warp) and reviews outside fixes; NVIDIA merges
portability-neutral fixes (see nvidia/warp PR #1702). `rocm-117` sits on AMD's
`amd-integeration-dev` (~17 commits behind nvidia/warp main as of 2026-08-13) — the 430-commit
sync gap is closed. Our fix stack on top (~10 commits) is all upstream candidates: the
`CUDA_CALLABLE` annotations and the cluster arch-string fix are NVIDIA-neutral
(nvidia/warp); capture enablement, the alloc guard, capture-safe LBVH rebuild, kernel-only
capture, and the test gatings belong in AMD-Ecosystem/warp; the mujoco_warp scratch cache
and island determinism belong in google-deepmind/mujoco_warp. Upstreaming early keeps the
branch small and rebaseable against AMD's fast-moving dev branch.

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
