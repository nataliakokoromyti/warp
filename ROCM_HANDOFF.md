# Warp + mujoco_warp on MI350X — collaborator handoff

*Goal: make Warp (and by extension mujoco_warp) fully validated and **super optimized** on
AMD Instinct MI350X (gfx950). Status as of 2026-08-17.*

## What this branch is

**`rocm-117` (current)** = [AMD-Ecosystem/warp](https://github.com/AMD-Ecosystem/warp)
`amd-integration-dev` (AMD's re-port of Warp onto near-current upstream — **Warp
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
  speedup (on the 1.13 branch: 2.94 graph / 4.2 eager). Captured step graph is 288 nodes,
  100% kernel nodes.
- **Full benchmark suite re-run on 1.17 with warm capture** (2026-08-17): g1_flat 1.51M
  steps/s, franka 5.08M, humanoid 1.40M — see the cross-vendor section for the full table
  and for why the previously reported gaps were mostly a cold-capture artifact.

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

Remaining perf levers, in evidence order (see the cross-vendor section for the data):
**(1) conditional graph nodes** — AMD-blocked, worth ~2-4× on solver-heavy scenes since we
replay a fixed 10 solver iterations where NVIDIA exits at ~3; **(2) collision kernels** —
`aloha_sdf` and `unitree_g1_hfield` are the only scenes warm capture did not help, so their
cost is collision compute; **(3) the 7 residual warm-graph allocations** on hfield.
Already tried and refuted, with data: MFMA/rocWMMA tile matmul (no gain at mujoco's 16x16
tiles) and wave-aware block sizing (slower). The scratch cache is a strong upstream
candidate for google-deepmind/mujoco_warp (CUDA graphs carry the same alloc nodes — 34 of
them on the L40S run — it simply costs CUDA far less); the kernel-only capture and the
warm-capture harness fix are both broadly useful.

## Cross-vendor benchmark comparison (updated 2026-08-17, rocm-117)

**Headline: most of the reported gap was benchmark methodology, not silicon.**
`mjwarp-testspeed` captured its graph **cold** (on a `Data` that had never stepped), so all
per-step scratch allocations became `memAlloc` nodes replayed on every launch. CUDA absorbs
that almost for free; ROCm does not. Measured on the same scene, same worlds, by us:

| g1_flat @8192 | MI350X (rocm-117) | L40S (stock warp 1.16, pristine mjw) |
|---|---|---|
| eager | 5.56 ms | 6.11 ms |
| cold-captured graph | 11.21 ms | 3.98 ms |
| warm-captured graph | 5.09 ms | 3.76 ms |
| **cold/warm penalty** | **2.20x** | **1.06x** |

Node census proves the mechanism: our cold graph = 288 kernel + **34 memAlloc** nodes; our
warm graph = 288 kernel, **zero** alloc nodes. So **graph-captured allocation nodes cost
CUDA ~6% and ROCm ~120%** -- the single most quotable number for AMD (see bug list below).

Re-running the L40S side with **our compat patch applied** (it is HIP-conditional, so on
CUDA only the vendor-neutral parts activate) gives a clean same-source hardware comparison:

| @8192, same source | MI350X | L40S | ratio |
|---|---|---|---|
| g1_flat eager | 5.56 ms | 5.45 ms | 1.02x (parity) |
| g1_flat warm-graph | 5.09 ms | 3.79 ms | **1.34x** |
| hfield eager | 9.60 ms | 53.27 ms | 0.18x (**AMD 5.6x faster**) |
| hfield warm-graph | 7.08 ms | 6.72 ms | 1.05x |

Two things this run also proves: the **scratch cache is vendor-neutral** (the L40S warm
graph drops from 34 memAlloc nodes to zero with our patch -- a real upstream contribution,
CUDA just suffers less without it), and the **7 residual warm-capture allocations on hfield
appear on BOTH vendors**, so that is a mujoco_warp bug rather than a ROCm one.

> **Caveat on the two-scene table above**: `cold_vs_warm.py` steps without a control
> trajectory, so the robot settles. The benchmark suite replays `shuffle_dance.npz` with
> ctrl noise. For g1_flat the workloads agree (5.09 vs 5.43 ms, 1.07x) so its numbers are
> representative; for **hfield the replay workload is 7.2x more expensive** (50.8 vs
> 7.1 ms) because the robot walks over terrain generating far more contacts. Treat the
> hfield "parity" as valid for the settled workload only -- the sweep table below is the
> representative comparison.

Fixing capture warmth (3 warmup steps before capture, now in the compat patch's
`cli.unroll`) moves the whole suite:

| scene | cold sweep | **warm sweep** | gain | gap vs NVIDIA (cold -> warm) |
|---|---|---|---|---|
| unitree_g1_flat | 601,815 | **1,507,513** | 2.50x | 4.2x -> **1.7x** |
| franka_emika_panda | 2,409,265 | **5,077,361** | 2.11x | 9.7x -> 4.6x |
| humanoid | 700,167 | **1,397,633** | 2.00x | 8.1x -> 4.1x |
| aloha_pot | 399,953 | 605,147 | 1.51x | 6.3x -> 4.2x |
| myoarm | 260,747 | 373,955 | 1.43x | 5.3x -> 3.7x |
| three_humanoids | 356,869 | 502,256 | 1.41x | 3.0x -> 2.1x |
| unitree_g1_hfield | 129,374 | 161,122 | 1.25x | 13.0x -> 10.4x |
| aloha_clutter | 59,213 | 68,987 | 1.17x | 6.1x -> 5.2x |
| mug | 148,861 | 155,746 | 1.05x | 3.1x -> 3.0x |
| aloha_sdf | 32,839 | 33,623 | 1.02x | 12.5x -> 12.2x |
| unitree_g1_hfield_render | 59,910 | 60,850 | 1.02x | 2.9x -> 2.8x |

Geomean **1.43x** throughput from the warmth fix alone.

### The definitive comparison: same source, same harness, both measured by us

L40S (stock warp 1.16 + our compat patch) vs MI350X (rocm-117 + same patch), warm capture
on both, identical scenes and world counts:

| scene | MI350X | L40S | L40S/AMD |
|---|---|---|---|
| unitree_g1_flat | 1,507,513 | 2,119,598 | **1.41x** |
| three_humanoids | 502,256 | 903,517 | 1.80x |
| unitree_g1_hfield_render | 60,850 | 161,748 | 2.66x |
| mug | 155,746 | 440,432 | 2.83x |
| myoarm | 373,955 | 1,235,704 | 3.30x |
| aloha_pot | 605,147 | 2,354,840 | 3.89x |
| humanoid | 1,397,633 | 5,474,229 | 3.92x |
| franka_emika_panda | 5,077,361 | 22,208,930 | 4.37x |
| aloha_clutter | 68,987 | 346,813 | 5.03x |
| unitree_g1_hfield | 161,122 | 1,040,014 | 6.45x |
| aloha_sdf | 33,623 | 378,106 | **11.25x** |
| **geomean** | | | **3.65x** |

That is the honest number to quote: **3.65x geomean behind an L40S** (a workstation-class
Ada card) on identical software. The gap concentrates in two buckets, both explained:
collision-heavy scenes (sdf, hfield, clutter) and small-`nv` solver scenes (franka nv=9,
humanoid nv=27, pot) where NVIDIA's conditional-node early exit means it runs ~1-3 solver
iterations to our fixed 10.

### Quantified: what conditional graph nodes are actually worth (2026-08-17)

Capping `m.opt.iterations` bounds what a working early exit would buy, because the HIP path
unrolls the full budget into the graph while NVIDIA exits at convergence. Measured
warm-graph, MI350X (`rocm-tools/iter_ceiling.py`):

| scene | scene's configured iterations | measured `niter_mean` | default | iterations=1 | **ceiling** |
|---|---|---|---|---|---|
| franka_emika_panda | **100** | 1.00 | 6.479 ms | 1.530 ms | **4.24x** |
| humanoid | **100** | 1.00 | 4.767 ms | 1.409 ms | **3.38x** |
| unitree_g1_flat | 10 | 1.00 | 5.297 ms | 4.650 ms | 1.14x |

The franka and humanoid benchmark scenes configure a budget of **100 solver iterations and
converge in 1** — so on HIP we replay ~99 iterations of dead work every step, while CUDA's
conditional node skips them for free. That is 4.24x and 3.38x left on the table, and it
lines up almost exactly with those scenes' 4.37x and 3.92x gaps vs the L40S in the table
above. **This is the strongest possible evidence for prioritizing hipGraph conditional
node support with AMD**, and it also bounds any home-grown workaround (a host-side
chunked-solver stepper: launch K iterations, read `nsolving`, repeat — costs 1-2 syncs per
step, which is <1% of a 6.5 ms step).

### Our mujoco_warp patch is safe on NVIDIA (2026-08-17)

Evidence for upstreaming the vendor-neutral parts (per-step scratch cache, EPA scratch
hoist, warm capture in `cli.unroll`). Full mujoco_warp suite on an L40S, pristine upstream
vs the same tree with our compat patch applied:

| | result |
|---|---|
| pristine upstream | 1241 passed, 22 skipped |
| **with our patch** | **1241 passed, 22 skipped** |
| **tests failing only with our patch** | **none** |

So the patch is a no-op for CUDA correctness while removing 34 alloc nodes from the L40S
warm-captured graph.

**Do not quote suite wall-clock as a patch speedup.** The first A/B appeared to show the
patch making the suite 2.84x faster (16:44 pristine vs 5:53 patched). It does not: the two
runs shared a per-job Warp kernel cache, so the second run inherited warm JIT artifacts. A
reversed-order control proved it -- whichever suite ran *second* took ~5.5 min either way:

| run order | first | second |
|---|---|---|
| pristine -> patched | pristine 16:44 | patched **5:53** |
| patched -> pristine | patched 11:27 | pristine **5:24** |

Suite wall-clock here measures JIT compilation, not physics. The patch's real perf effect
is in the graph-node counts and per-step timings elsewhere in this document.

### Prototype: recovering the conditional-node win without AMD (2026-08-17)

`rocm-tools/chunked_stepper.py` emulates conditional-graph early exit with host-side
control. It splits the step into three captured graphs -- **pre** (forward dynamics +
solver init), **chunk** (K solver iterations), **post** (solver tail + sensor_acc +
integrator) -- and loops on the chunk graph from the host, reading the 1-int `nsolving`
counter between chunks to stop at convergence. The seam is clean because `_solve`
decomposes into init / loop / one conditional tail launch, and `m.opt.iterations = 0`
makes it skip exactly the loop; the solver context it needs already persists on `Data`
thanks to the scratch cache.

Validated on both vendors (chunk=1; chunk=2 is slightly worse since these scenes converge
in one iteration):

| scene | MI350X reference | MI350X chunked | **speedup** | ceiling | L40S speedup |
|---|---|---|---|---|---|
| franka | 6.546 ms | **1.688 ms** | **3.88x** | 4.24x | 1.00x |
| humanoid | 4.703 ms | **1.573 ms** | **2.99x** | 3.38x | 1.00x |
| unitree_g1_flat | 5.540 ms | 5.319 ms | 1.04x | 1.14x | 1.00x |

Each lands just under its measured ceiling -- the shortfall is the per-chunk host sync --
and costs nothing on CUDA, where conditional nodes already do this in hardware.

Faithfulness, checked against unmodified `mjw.step` over 10 steps from an aligned start,
with a control of a second independent `mjw.step` run (MI350X):

| scene | chunked vs ref | ref vs ref (control) |
|---|---|---|
| franka | **0.000e+00** (bit-identical) | 0.000e+00 |
| humanoid | 6.079e-07 | 5.839e-07 |
| unitree_g1_flat | 2.327e-06 | 2.428e-06 |

franka is exact; the others sit inside mujoco_warp's own run-to-run nondeterminism. Same
picture on the L40S (4.5e-13 / 5.1e-07 / 2.4e-06 against controls of the same magnitude).

**Projected suite impact** (applying the measured per-scene speedups to the warm sweep --
a projection, the stepper is not wired into `testspeed`): franka 5.08M -> ~19.7M steps/s
vs L40S 22.2M (**1.13x**), humanoid 1.40M -> ~4.18M vs 5.47M (**1.31x**), g1_flat 1.51M ->
~1.57M vs 2.12M (1.35x). That would put these three scenes near parity with the L40S,
versus 4.37x / 3.92x / 1.41x today.

**Status**: prototype, not landed in the library. Productionizing means deciding how users
opt in (mujoco_warp's `step()` is monolithic by design), and the AMD-side timing rerun with
the control is still queued. Known constraint, enforced by an assert: the stepper requires a
warmed `Data` -- built on a cold one, the `nsolving` counter is allocated *inside* the
capture as a graph allocation and cannot be read from the host (the failed read also
poisons the HIP context and aborts the process).

**Corrected long-standing issue**: the cloth family (`cloth`, `cloth_render`,
`aloha_cloth`) overflows on the **L40S too** (22/31/31 worlds vs our 26/29/29) with the
same assets and settings. The old handoff item claiming AMD uniquely needs `nconmax~26000`
was wrong -- this is a mujoco_warp/scene-config issue, not a ROCm accounting difference.
`primitives` runs on the L40S (1.19M steps/s) but still fails on AMD -- that one *is*
ours to triage.

Where the remaining gap actually lives, now that allocation noise is gone:

1. **Conditional graph nodes (still AMD-blocked).** The L40S graph has **144 kernel nodes
   plus a conditional node**; ours has **288 unrolled** kernel nodes. NVIDIA's `capture_while`
   exits the solver at convergence (~3 iterations on G1); we must replay the full fixed
   budget (10). This is structural, not tuning, and only AMD can unblock it.
2. ~~**Collision-dominated scenes.**~~ — **retired 2026-08-18**, see
   "Collision compute is an AMD strength" below. Isolated per-phase timings on both
   vendors show collision is *faster* on MI350X in every scene except one, and the sole
   exception is a single kernel (`_sdf_narrowphase`) with a fixable launch configuration.
3. ~~Residual warm-graph allocations on hfield~~ — **fixed 2026-08-17**: traced to 7 real
   allocations in `convex_narrowphase` (the EPA polytope scratch + ccd counter, sized from
   `naccdmax`; 11 further calls are zero-sized and allocate nothing). Hoisted into the
   `Data` scratch cache, so the hfield warm graph is now **291 nodes, 100% kernel, zero
   memAlloc** and 15% faster (7.08 -> 5.99 ms; 1.16M -> 1.37M steps/s). These fired on the
   L40S too, so it is an upstream mujoco_warp win, not an AMD workaround.

**Refuted -- do not retry without new evidence:** wave-aware block sizing. mujoco_warp
derives solver widths as `clamp(round_up_32(nv), 32, 256)` and keeps eight static 32-wide
defaults, which look wrong on 64-lane wavefronts, and the scenes rounding to 32 (franka
nv=9, humanoid nv=27, clutter nv=22) were exactly the worst-gap scenes. Measured, it is
wrong: on franka every wave-aligned override was **slower** (`linesearch_iterative=64` by
30%, full wave-aware set by 23%); humanoid and g1_flat moved +1-2%, inside noise. Reason:
`launch_tiled(dim=nworld)` creates one block per world regardless of block size, so raising
`block_dim` does not fill idle wavefronts -- it assigns more threads to tiles only `nv`
wide, while reducing blocks resident per CU. Tool: `rocm-tools/blockdim_tune.py`.

### Collision compute is an AMD strength -- except one kernel (2026-08-18)

`aloha_sdf`, `unitree_g1_hfield` and `aloha_clutter` were filed as "collision-dominated"
because warm capture did not help them. That inference does not hold: warm capture also
does not help a scene whose cost is the *unrolled solver*, and end-to-end steps/s cannot
tell the two apart. `rocm-tools/collision_bench.py` measures the collision pipeline on its
own -- it steps a scene into a representative state, then times `collision(m, d)` in
isolation and subtracts a run with each narrowphase stubbed out. Same source, same scenes,
same states, run on both vendors:

| isolated phase, ms/call | MI350X | L40S | L40S / MI350X |
|---|---|---|---|
| hfield, whole collision pipeline | 3.469 | 46.944 | **0.074x (AMD 13.5x faster)** |
| hfield, convex narrowphase (CCD) | 3.189 | 46.564 | **0.068x (AMD 14.6x faster)** |
| clutter, whole collision pipeline | 1.618 | 3.507 | 0.46x (AMD 2.2x faster) |
| clutter, convex narrowphase | 0.947 | 2.413 | 0.39x (AMD 2.5x faster) |
| g1_flat, whole collision pipeline | 0.547 | 0.815 | 0.67x (AMD 1.5x faster) |
| **aloha_sdf, `_sdf_narrowphase`** | **30.248** | **1.233** | **24.5x (AMD slower)** |

So the CCD/GJK/EPA stack -- the thing wave64 divergence was supposed to punish -- is
comfortably faster on MI350X, and the broadphase is small either way (AMD is ~3.5x slower
on the tiny nxn broadphase, 0.19 vs 0.055 ms on hfield: real, but 0.1 ms). Every scene
except `aloha_sdf` needs to be re-attributed to the solver, not to collision.

The same conclusion from the other direction, using `-o opt.iterations` as a control
(MI350X, `rocm-tools/slurm/col_iterceil.sbatch`; both scenes configure a **100**-iteration
budget and converge in 1-3):

| scene | default | iterations=4 | iterations=1 | reading |
|---|---|---|---|---|
| aloha_sdf | 38,806 / 38,563 | 36,009 | 29,016 | solver is ~free; **collision is the cost** |
| aloha_clutter | 87,338 / 86,283 | -- | 126,854 | **1.45x** sits in the unrolled solver |

(Caveat: neither scene replays a control trajectory in this test, so cutting iterations
changes the physics -- both vendors get *slower* at `iterations=1` because the sim stops
converging. The same reversal appears on the L40S, so the comparison is still like-for-like;
just do not read `iterations=1` as a ceiling.)

Where `aloha_sdf`'s time actually goes, from mujoco_warp's own event trace
(`rocm-tools/etrace_agg.py`, steady-state eager step, MI350X):

| scope | MI350X | L40S |
|---|---|---|
| `step` | 34.83 ms | -- |
| `forward.fwd_position.collision.sdf_narrowphase` | **30.25 ms (87% of the step)** | 1.23 ms |
| `...collision.convex_narrowphase` | 0.34 | 1.45 |
| `...collision.primitive_narrowphase` | 0.04 | 0.05 |
| `...collision.nxn_broadphase` | 0.06 | 0.03 |
| `forward.solve` | 1.39 | 12.21 |

One kernel, 87% of the step, 24.5x off the L40S. That is the entire `aloha_sdf` gap.

**Root cause: the kernel had no `__launch_bounds__`.** Without it the HIP compiler must
assume the maximum flat workgroup size (1024 threads = 16 waves resident on one CU = 4
waves per SIMD), which caps the kernel at 128 VGPRs. `_sdf_narrowphase` inlines the whole
gradient-descent / Wolfe line-search / octree-sampling stack into one body and wants far
more than that, so it spills to scratch. Declaring the true block size lifts the cap.
Measured on `aloha_sdf` @8192, warm graph, controls interleaved
(`rocm-tools/slurm/col_sdf_sweep.sbatch`):

| config (block_dim, `__launch_bounds__`) | steps/s |
|---|---|
| stock (256, none) -- three controls | 32,955 / 35,779 / 33,630 |
| **256 + `__launch_bounds__(256, 1)`** | **55,269 (1.61x)** |
| 128 + `__launch_bounds__(128, 1)` | 33,365 |
| 64 + `__launch_bounds__(64, 1)` | 35,724 |
| 512 + `__launch_bounds__(512, 1)` | 33,583 |
| 256 + `__launch_bounds__(256, **2**)` | 33,551 |

Two things worth reading off this table. The winning row launches at Warp's *default* block
size, so the only difference from the control is the presence of the attribute -- nothing
about the launch geometry changed. And on HIP the second `__launch_bounds__` argument is
MIN_WARPS_PER_EXECUTION_UNIT, so `2` halves the register budget to 256 VGPRs: it erases the
entire win, which pins the kernel's requirement at **>256 VGPRs** against an implicit cap of
128. (Same trap as the `_CCD_MIN_BLOCKS=8` finding on the hfield CCD kernel -- NVIDIA's
min-blocks-per-SM semantics do not carry over.)

The compiled binaries say it outright. Reading the AMDGPU metadata note out of the two
`.cubin`s Warp cached for this kernel (`rocm-tools/hsaco_regs.py`):

| | stock | `__launch_bounds__(256, 1)` |
|---|---|---|
| `max_flat_workgroup_size` | 1024 | 256 |
| `vgpr_count` | **128** (the cap) | **465** |
| `agpr_count` | 0 | 209 |
| **`vgpr_spill_count`** | **465** | **0** |
| `sgpr_spill_count` | 125 | 95 |
| `private_segment_fixed_size` (scratch) | 21,088 B | 19,104 B |

465 spilled VGPRs, in the innermost loop of a gradient descent that re-samples an octree
ten times per iteration. Declaring the bound moves all of them back into registers.

**This generalises past mujoco_warp, and the fix belongs in Warp.** Warp compiles a module
once per `block_dim` and launches it at exactly that width, so it always knows the bound at
codegen time and simply never emitted it. `warp/_src/codegen.py` now emits
`WP_DEFAULT_LAUNCH_BOUNDS` for kernels that declare none; the macro expands to
`__launch_bounds__(WP_TILE_BLOCK_DIM)` under `__HIPCC__` and to nothing otherwise, so
NVIDIA codegen is semantically unchanged. A/B'd against **stock, unpatched mujoco_warp**,
two full passes with a separate kernel cache per variant
(`rocm-tools/slurm/col_warplb.sbatch`):

| scene, steps/s | off (pass 1 / 2) | on (pass 1 / 2) | ratio |
|---|---|---|---|
| aloha_sdf | 39,859 / 39,811 | **56,816 / 56,657** | **1.43x** |
| aloha_clutter | 88,035 / 86,428 | 85,788 / 86,876 | 0.99x |
| unitree_g1_hfield | 897,540 / 893,224 | 901,664 / 893,297 | 1.00x |

Pass-to-pass spread is under 2%, so both the sdf gain and the two non-regressions are real,
and no mujoco_warp change is needed. The fix's reach across the rest of the suite is narrow
but its downside is nil: of 40 code objects in an `aloha_sdf` kernel cache 8 spill, and
after `_sdf_narrowphase`'s 465 the next worst are `linesearch_iterative` (10) and
`primitive_narrowphase` (7). It removes a cliff rather than lifting a floor.

Also refuted this round: `rocprofv3 --kernel-trace` is unusable on a captured mujoco_warp
run -- it hangs on `aloha_sdf` and segfaults with `--output-format csv` (matching the known
`--stats` hang). Use `collision_bench.py` / the event trace instead.

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
- ~~Cloth benchmarks need `nconmax≈26000` vs the NVIDIA-tuned 2,200~~ — **retired
  2026-08-17**: the same overflow reproduces on an NVIDIA L40S with identical assets and
  settings, so it is not an AMD accounting difference. Upstream/scene issue.

## ROCm bugs worth filing with AMD

**See `AMD_ROCM_ISSUES.md`** — a staged, self-contained report with environment, measured
numbers, repro scripts and root-cause reads (not yet sent to AMD). Summary of what it
covers, in value order: (1) hipGraph **conditional node** support, worth a measured 4.24x
on franka / 3.38x on humanoid; (2) graph **allocation nodes replay ~20x worse than CUDA**
(2.20x vs 1.06x cold/warm penalty); (3) the **stuck invalidated capture** bug, already
fixed on clr `develop` as `fa77aed` but unreleased — ask for a 7.2.x backport;
(4) minor CUDA-semantics divergences. Original working notes below.

### Original working notes

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
`amd-integration-dev` (~17 commits behind nvidia/warp main as of 2026-08-13) — the 430-commit
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
