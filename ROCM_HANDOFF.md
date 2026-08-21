# Warp + mujoco_warp on MI350X — collaborator handoff

*Goal: make Warp (and by extension mujoco_warp) fully validated and **super optimized** on
AMD Instinct MI350X (gfx950). Status as of 2026-08-18.*

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
**(1) conditional graph nodes** — was AMD-blocked and worth ~2-4× on solver-heavy scenes;
**recovered in software 2026-08-18 by `mjw.Stepper`**, see below; **(2) `_sdf_narrowphase`** —
was the one collision kernel slower on AMD, **fixed 2026-08-18** (unreachable device printf +
static octree index: 14.0× on the kernel, 2.20× end-to-end on aloha_sdf), leaving an
as-yet-unattributed 3.92× residual vs the L40S; the rest of the collision stack is *faster*
than an L40S, so hfield and clutter belong under lever 1; ~~**(3) the 7 residual warm-graph
allocations** on hfield~~ — fixed.
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

### Recovering the conditional-node win without AMD: `mjw.Stepper` (2026-08-18)

**Status: implemented in `patches/mujoco_warp-chunked-stepper.patch`, validated on both
vendors, and shaped as an upstream PR to google-deepmind/mujoco_warp.** It supersedes the
`rocm-tools/chunked_stepper.py` prototype (kept for reference).

The idea is unchanged: split the step into three captured graphs -- **pre** (forward
dynamics + solver init), **chunk** (K solver iterations), **post** (solver tail +
sensor_acc + integrator) -- and drive the middle one from the host, reading the 1-int
`nsolving` counter between batches to stop at convergence. What changed is how the split
is produced and how it fails.

**The split is recorded, not re-derived.** The prototype reimplemented `forward()` by
copy-paste, so it silently did the wrong thing on sleep, islands, the compact solve and
user callbacks. Instead, `solver._solve` now calls the loop through a one-line indirection
(`_run_solver_iterations`), and `Stepper` installs an *iteration driver* into it while it
captures one real `mjw.step(m, d)`. When the recorded step reaches the solver loop the
driver ends the capture (that is `pre`), captures a batch of iterations on its own, and
opens a fresh capture for the rest of the step (`post`). Nothing about `step()` is
duplicated, so every configuration it supports is supported by construction, and `step()`
itself is untouched.

**It degrades to today's behavior.** `Stepper.step()` is always semantically
`mjw.step(m, d)`; the only question is how many graphs it is. It declines the split and
captures one monolithic graph -- exactly what a caller writes by hand today -- when
conditional graph nodes exist (CUDA), for RK4 (four solver loops per step), for
`iterations < 2`, on a `Data` that has never been stepped, or if the recording finds a
number of solver loops other than one. It also abandons the split at run time if the first
eight steps all used more than half their iteration budget, since then there is little dead
work to skip; it captures the monolithic graph lazily at that point and switches.
`stepper.split` and `stepper.reason` report which and why; `split=True` turns the decline
into an error (and pins the split) for tests and benchmarks. A declined recording rolls
back everything it allocated, so the fallback graph is not poisoned by buffers belonging to
a discarded capture.

**Batch size adapts.** Chunk graphs are captured at sizes 1, 2, 4, ... 32; each step starts
from the batch that covered the previous step and doubles, clamped so it never overshoots
the iteration budget. A solve needing *n* iterations costs O(log n) host reads, and a scene
with a stable iteration count settles on one -- or zero, when the batch already covers the
whole budget and there is nothing left to skip. `chunk=K` pins a fixed size instead.

**Supporting fixes it needed** (all also remove per-step allocation nodes on both vendors):
the compact/sleep solve rebuilt its `SolverContext` and its `nsolving` counter on every
step, because it runs on a `dataclasses.replace` copy of `Data` that the caches could not
survive; both are now anchored on the real `Data`. Without this the sleep path's chunk
graph would reference buffers freed since capture. The sparse-flex diagonal preconditioner
was likewise allocated per solve and read by every iteration, so it is now Data-cached.

Measured on MI350X, warm capture, adaptive batch, **auto policy (no forcing)** -- reference
is the monolithic warm-captured graph, the current best practice
(`rocm-tools/stepper_validate.py`, job `amd-16888525`):

| scene | reference | Stepper | **speedup** | iters/budget | host reads/step |
|---|---|---|---|---|---|
| franka_emika_panda | 6.519 ms | **1.711 ms** | **3.81x** | 2 / 100 | 1 |
| humanoid | 4.880 ms | **1.813 ms** | **2.69x** | 6 / 100 | 2 |
| aloha_pot | 17.799 ms | **10.801 ms** | **1.65x** | 2 / 100 | 1 |
| **aloha_clutter (SLEEP, 22 trees, compact solve)** | 20.461 ms | **12.498 ms** | **1.64x** | 1 / 100 | 1 |
| three_humanoids | 12.370 ms | **7.597 ms** | **1.63x** | 3 / 100 | 2 |
| unitree_g1_flat | 5.558 ms | 5.617 ms | 0.99x | 8 / 10 | 2 |
| unitree_g1_hfield | 10.371 ms | 10.461 ms | 0.99x | 10 / 10 | — (abandoned) |

The gain tracks exactly how much of the budget the solve leaves unused -- the quantity a
conditional node exploits. franka lands just under its independently measured 4.24x
ceiling; the shortfall is the host read. The two scenes at 0.99x are the honest floor:
their 10-iteration budget is about right-sized, so there is almost nothing to skip. hfield
uses all 10 and the run-time calibration abandons the split for it; g1_flat uses 8 of 10,
which is enough early exit to keep the split but not enough to pay for it.

**End-to-end through `testspeed`** (1000 steps, `--chunked_solver` on vs off, MI350X) --
this replaces the earlier projection with measured suite numbers:

| scene | chunked off | **chunked on** | ratio | L40S (same harness) | gap before | **gap now** |
|---|---|---|---|---|---|---|
| franka_emika_panda | 5,028,269 | **19,301,515** | **3.84x** | 22,208,930 | 4.37x | **1.15x** |
| humanoid | 1,410,934 | **2,740,697** | **1.94x** | 5,474,229 | 3.92x | **2.00x** |
| unitree_g1_flat | 1,658,881 | 1,597,235 | 0.96x | 2,119,598 | 1.41x | 1.33x |

franka goes from 4.37x behind an L40S to **1.15x**, humanoid from 3.92x to 2.00x. g1_flat
loses 4%.

On CUDA the auto policy declines the split, so it is a measured **1.00x on all seven
scenes** -- the feature cannot regress a platform that already has conditional nodes.

To test the *decomposition* rather than the platform, the same scenes ran on an L40S with
`opt.graph_conditional` forced off, which makes CUDA unroll the budget exactly as HIP does
(`STEPPER_UNROLL=1`). This is the apples-to-apples check, on a quiet node, and it covers the
paths the prototype could not reach:

| scene (L40S, unrolled reference) | reference | Stepper | speedup |
|---|---|---|---|
| franka_emika_panda | 6.896 ms | 1.590 ms | **4.34x** |
| humanoid | 4.216 ms | 1.009 ms | **4.18x** |
| aloha_pot | 15.556 ms | 4.851 ms | **3.21x** |
| **aloha_clutter (SLEEP, 22 trees, compact solve)** | 11.021 ms | 4.087 ms | **2.70x** |
| three_humanoids | 11.653 ms | 6.577 ms | 1.77x |
| unitree_g1_flat | 4.844 ms | 4.689 ms | 1.03x |
| unitree_g1_hfield | 8.886 ms | 9.326 ms | 0.95x (abandoned mid-measurement) |

**Faithfulness.** Every scene is checked against unmodified `mjw.step` over 10 steps from
an aligned start, against a control of a second independent `mjw.step` run -- the only
meaningful yardstick, since mujoco_warp's constraint assembly uses atomics and diverges
run-to-run on its own. On MI350X franka is **bit-identical (0.000e+00, control 0.000e+00)**
and every other scene's stepper delta sits at its control:

| MI350X, 10 steps | stepper vs ref | control (ref vs ref) |
|---|---|---|
| franka_emika_panda | **0.000e+00** | 0.000e+00 |
| aloha_pot | 1.049e-05 | 1.013e-05 |
| aloha_clutter | 7.057e-05 | 6.962e-05 |
| humanoid | 3.906e-03 | 3.891e-03 |
| three_humanoids | 4.440e-03 | 4.120e-03 |
| unitree_g1_flat | 4.669e-03 | 4.837e-03 |
| unitree_g1_hfield | 2.620e+03 | 2.773e+03 |

(hfield's absolute numbers are large because the scene is chaotic over 10 steps -- the
control is the point, not the magnitude.) Same picture on the L40S in both modes. A probe
(`rocm-tools/stepper_niter_probe.py`) confirmed the mechanism directly: a Stepper pinned to
`chunk=iterations`, which runs *exactly* what the unrolled reference runs, differs from the
reference by the same amount as `chunk=1` does. The residual is nondeterminism, not the
split.

`mujoco_warp/_src/stepper_test.py` covers trajectory equivalence (default, sleep+islands,
Euler/implicitfast, fixed chunk, capturing `forward` instead of `step`), early exit, the
calibration hand-off, and every fallback: **16 passed on the L40S** (three consecutive runs,
no flakes), **15 passed + 1 principled skip on MI350X** (the conditional-graph fallback test
has nothing to assert on HIP). The modules the solver refactor touches (solver, sleep,
forward, island): **215 passed on the L40S, 211 passed + 4 skips on MI350X**, zero failures.
The **full mujoco_warp suite on the L40S with both patches applied: 1,257 passed, 22
skipped, 0 failed** -- pristine upstream is 1,241 passed / 22 skipped, so that is the same
1,241 plus the 16 new tests.

**Known gaps.** RK4 falls back (it runs four solver loops; a multi-loop split is possible
but was not built). The calibration's threshold is a heuristic on iteration counts, not a
timing measurement, so it catches the clear case (hfield, 10/10) and not the marginal one
(g1_flat, 8/10, which costs 4% end to end on MI350X and *gains* 3% on the L40S -- the sign
is platform-dependent, which is why a fixed heuristic cannot get both). Constructing a
Stepper advances the Data by `warmup` steps; there is no snapshot/restore.

**Opt-in.** `mjw.Stepper(m, d)` -- an object, not a flag, because a captured stepper owns
graphs with a lifetime and `step()` is deliberately monolithic. Constructing it advances
`d` by `warmup` steps (default 3) to force the lazy allocations that capture must not make;
`warmup=0` for an already-stepped `Data`. `fn=` captures something other than `step`
(`forward`, `step2`). `cli.unroll` uses it automatically (`--chunked_solver`, default on),
so `testspeed` and the benchmark suite pick the win up without changes.

**Where it belongs: upstream.** The change is vendor-neutral and additive -- `step()` is
unchanged, the loop indirection is pure code motion, and on CUDA the policy declines the
split. It is not only an AMD workaround: `m.opt.graph_conditional=False` is also the
documented JAX path, and those users eat the full unrolled budget on CUDA today. Carrying
it in our patch stack indefinitely means rebasing a 790-line diff against a fast-moving
upstream.

**Corrected long-standing issue**: the cloth family (`cloth`, `cloth_render`,
`aloha_cloth`) overflows on the **L40S too** (22/31/31 worlds vs our 26/29/29) with the
same assets and settings. The old handoff item claiming AMD uniquely needs `nconmax~26000`
was wrong -- this is a mujoco_warp/scene-config issue, not a ROCm accounting difference.
`primitives` runs on the L40S (1.19M steps/s) but still fails on AMD -- that one *is*
ours to triage.

Where the remaining gap actually lives, now that allocation noise is gone:

1. ~~**Conditional graph nodes (AMD-blocked).**~~ — **largely recovered 2026-08-18** by
   `mjw.Stepper` (section above): host-driven solver batches, at the cost of ~1 host read
   per step. Still worth asking AMD for real conditional nodes -- they would remove the
   reads entirely and the ~1-2% they cost on scenes that use their whole iteration budget --
   but this is no longer a blocker.
2. ~~**Collision-dominated scenes.**~~ — **retired 2026-08-18**, see "Collision compute is
   an AMD strength" below. Isolated per-phase timings on both vendors show collision is
   *faster* on MI350X in every scene except one, and that sole exception
   (`_sdf_narrowphase`) is now fixed.
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
| `forward` | 32.68 | 16.11 |
| `forward.fwd_position.collision.sdf_narrowphase` | **30.25 (87% of the step)** | 1.23 |
| `...collision.convex_narrowphase` | 0.34 | 1.45 |
| `...collision.primitive_narrowphase` | 0.04 | 0.05 |
| `...collision.nxn_broadphase` | 0.06 | 0.03 |
| `forward.solve` | 1.39 | 12.21 |

One kernel, 87% of the step, 24.5x off the L40S. That is the entire `aloha_sdf` gap. It has
two causes, and the smaller one is the one that looks like the obvious answer.

**First cause: the kernel had no `__launch_bounds__`.** Without it the HIP compiler must
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

**The bigger half of the kernel's cost is six device `printf`s.** `collision_sdf.py` carries
six `wp.printf` / `wp.print` error diagnostics inside `find_oct`, `sdf` and `sdf_grad` --
unreachable-error paths, buried under a deeply inlined call tree. On HIP `printf` lowers to
an OCKL **hostcall**, which the compiler must treat as an opaque, memory-clobbering call: it
materialises a varargs buffer at each site and forces live values across it. Six of them
inside the octree walk is what drives the register demand in the first place. Timing the
kernel on its own (`collision_bench.py`, isolated `sdf_narrowphase`, ms/call,
`rocm-tools/slurm/col_sdf_kernel_sweep.sbatch`):

| variant | stock body | prints removed |
|---|---|---|
| no `__launch_bounds__` (control x2) | 263.50 / 262.47 | **21.96 / 21.90** |
| `__launch_bounds__(64)` | 145.68 | 32.67 |
| `__launch_bounds__(256)` | 154.13 | 38.82 |
| `__launch_bounds__(512)` | 155.41 | -- |
| `__launch_bounds__(1024)` | **255.25** | -- |
| `__launch_bounds__(256, 2)` | 146.90 | -- |

Removing the prints is worth **12.0x on the kernel** -- against the L40S's 16.53 ms for the
same isolated kernel in the same harness, that moves MI350X from **15.9x slower to 1.33x
slower**. Repeat controls agree to 0.4%, and the `__launch_bounds__(1024)` row is the
internal control that pins the mechanism: 1024 *is* the value the compiler already assumed,
and declaring it explicitly buys nothing.

End to end it is **2.27x on `aloha_sdf`**, with the controls interleaved
(`rocm-tools/slurm/col_sdf_round4.sbatch`, @8192, 400 steps):

| run | steps/s | `ncon_mean` |
|---|---|---|
| stock a / b / c | 40,290 / 50,353 / 45,597 | 16.5138 / 16.5139 / 16.5144 |
| **prints removed a / b** | **100,753 / 105,486** | 16.5129 / 16.5131 |

`ncon_mean` agrees to five significant figures, and stock-to-stock varies by as much as
stock-to-patched, so the physics does not move. Two checks say the prints are safe to
compile out: they **never fire** (an unfiltered 512-world run emits zero `ERROR` lines, so
the 12x is entirely a compile-time effect and is not hiding a real octree failure), and
`nacon` after 20 steps differs between two *identical* stock runs by as much as it does
between stock and patched -- mujoco_warp's SDF scene is run-to-run nondeterministic on ROCm
at that horizon, which is why the equivalence check has to be made at 1-3 steps against a
same-horizon control.

The two fixes are not additive, they are alternatives, and the register data says why
(`rocm-tools/hsaco_regs.py`):

| build | max_flat_wg | vgpr | agpr | vgpr spills | scratch |
|---|---|---|---|---|---|
| stock | 1024 | 128 | 0 | **465** | 21,088 |
| stock + `__launch_bounds__(256)` | 256 | 465 | 209 | 0 | 19,104 |
| prints removed | 1024 | 128 | 0 | 93 | 8,820 |
| prints removed + `__launch_bounds__(256)` | 256 | 345 | 89 | 0 | 7,636 |

Once the prints are gone the kernel nearly fits in 128 VGPRs (93 spills), and raising the
cap then *costs* 1.8x: at 345+89 registers occupancy collapses to about one wave per SIMD,
and that is worse than paying for 93 cheap spills. So the launch-bounds default is a
guardrail against the cliff, not a free win everywhere -- worth keeping in mind if a future
Warp kernel regresses on ROCm.

**Third, smaller lever: the octree descent indexes a register vector dynamically.**
`find_oct` picks the next child with `oct_child[node][4*z + 2*y + x]`. AMD GPUs have no
indexed register-file access, so an 8-wide value indexed by a runtime value either goes to
scratch or becomes a select chain -- inside a 100-iteration pointer-chasing loop that is the
hottest code in the kernel. Writing the select chain out by hand
(`rocm-tools/sdf_oct_patch.py`, which also hoists the eight repeated `oct_child[node]` loads
in the leaf test into one) is worth little while `printf` dominates and becomes visible once
it does not (`rocm-tools/slurm/col_sdf_round3.sbatch`):

| body | no `__launch_bounds__` | `__launch_bounds__(256)` |
|---|---|---|
| stock (drift control x2) | 263.05 / 264.39 | 153.51 / 152.79 |
| + static octree index | 259.95 (1.2%) | 144.26 (6.0%) |
| + static octree index, prints removed | **18.84 (14% over prints-removed alone)** | 37.30 |

Stacked, `_sdf_narrowphase` goes **263.0 ms -> 18.84 ms, 14.0x** (see the closing table for
the L40S comparison). Repeat controls in this job agree to 0.5%, and the four measurements of
the stock configuration across three independent jobs landed on 263.50 / 262.47 / 263.05 /
264.39.

So the ranked fix list for the one genuinely AMD-hostile collision kernel is: compile out the
device prints (12.0x), then the static octree index (a further 1.16x), and `__launch_bounds__`
only matters if the prints stay.

**Validated in the shape it should be upstreamed** (`rocm-tools/slurm/col_sdf_round5.sbatch`).
The prints are *guarded*, not deleted -- `rocm-tools/sdf_debugprint_patch.py` wraps each in
`if _SDF_DEBUG_PRINT:` on a module constant, so Warp emits `if (false)` and the branch dies
in the first simplification pass while `MJW_SDF_DEBUG_PRINT=1` brings the diagnostics back.
It measures the same as deleting them:

| | isolated kernel, ms | `aloha_sdf` @8192, steps/s |
|---|---|---|
| stock (controls) | 235.48 / 236.35 | 44,439 / 49,557 |
| guarded prints | 22.05 | 74,179 |
| **guarded prints + static octree index** | **18.74** | **101,266 / 105,751** |

**2.20x end to end**, from 46,998 to 103,509 steps/s. (The 74,179 for guarded-alone is below
`round4`'s 100,753 / 105,486 for the same change with the prints deleted; the node was
contended and that single point should not be read as the guard being worse -- at kernel
level guarded is 22.05 against deleted's 21.96 / 21.90.)

Correctness, checked against a same-horizon control because the scene is chaotic and ROCm
run-to-run nondeterministic (`rocm-tools/sdf_equiv.py`, 512 worlds, 1 step):

| field | stock vs stock (control) | patched vs stock |
|---|---|---|
| `contact.dist` | 1.104892e-02 | 1.104894e-02 |
| `contact.pos` | 3.560256e-02 | 3.560257e-02 |
| `contact.frame` | 6.340905e-01 | 6.340907e-01 |
| `contact.geom` / `nacon` | 0 | 0 |
| `qpos` | 9.08e-09 | 1.55e-07 |
| `qvel` | 9.08e-06 | 1.55e-04 |

The contact arrays differ from stock by *the same amount two identical stock runs differ*:
contacts are appended in nondeterministic order (the `worldid` column permutes by ~500
either way), so those columns measure ordering, not physics. `nacon` and the geom pairing
are exact. State drift after one step is 1.6e-7 in `qpos` -- about 17x the run-to-run floor,
which is what changing register pressure and FMA contraction does to a chaotic contact
solve, not a semantic change.

Both test suites, each with its control run in the same job: the **Warp suite is 8,294 tests,
`OK (skipped=240)`** with the codegen change on and off
(`rocm-tools/slurm/col_validate_warp.sbatch`), and the **mujoco_warp suite is 1,233 passed /
30 skipped** with the SDF fixes and without (`rocm-tools/slurm/col_validate_mjw.sbatch`) --
both identical to the branch baseline. (The patched mujoco_warp run took 109 s against the
stock run's 886 s. That is the JIT-cache warming artifact documented above, *not* a speedup;
the two runs shared a kernel cache and the second inherited it.)

**Both mujoco_warp fixes are upstream candidates, not ROCm workarounds.** The same patched
source on an L40S (`rocm-tools/slurm/col_sdf_nv_ab.sbatch`) is slightly *faster*, never
slower:

| L40S | stock a | patched | stock b |
|---|---|---|---|
| isolated `_sdf_narrowphase`, ms | 16.452 | **16.048** | 16.490 |
| `aloha_sdf` @8192, steps/s | 398,803 | **405,298** | 398,240 |

+2.5% on the kernel and +1.6% end to end on CUDA. Which closes the loop on the whole
investigation:

| `_sdf_narrowphase`, isolated | MI350X | L40S | ratio |
|---|---|---|---|
| stock | 235.5 - 263.5 | 16.45 | **14.3 - 16.0x** |
| patched | **18.74** | 16.05 | **1.17x** |

`aloha_sdf` end to end goes from **8.48x behind the L40S to 3.92x**. The remaining 3.9x is
**not attributed yet** -- the kernel that used to be 87% of the step is now within 17% of the
L40S in the isolated harness, so whatever dominates the benchmark's (heavier, more-contact)
state is something else. Redo the phase split on the patched build before guessing; the
`opt.iterations` control is the obvious first suspect, since this scene budgets 100 solver
iterations with an elliptic cone and converges in ~3, and that control was previously
meaningless here only because collision swamped it.

**Next lead, unexplored:** mujoco_warp has **28 device prints** in total, and the file with
the most is `collision_flex.py` (**9**) -- the cloth family, which is separately known to be
among the worst-performing scenes on this branch. `smooth.py` (2), `forward.py` (4) and
`collision_convex.py` (3) are also worth an A/B. Anyone picking this up: apply
`rocm-tools/sdf_debugprint_patch.py`'s guard pattern to the file, then time with
`rocm-tools/collision_bench.py` and read `vgpr_spill_count` with
`rocm-tools/hsaco_regs.py` -- a spill count in the hundreds under
`max_flat_workgroup_size: 1024` is the signature.

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

- ~~Deterministic mode is not ported~~ — **fixed 2026-08-18**. Root cause was a
  preprocessor gap, not a wave64 issue: `warp/native/deterministic.h` guarded every
  device-side helper and every `WP_DET_*` macro on `__CUDA_ARCH__` alone, which HIPRTC
  never defines, so generated kernels always took the CPU fallback branch. The
  consumed-return counter fell back to a plain `wp::atomic_add` (scheduling-order slot)
  and the phase guards vanished, so phase 0 *and* phase 1 both mutated user state —
  `test_conditional_counter` returned exactly 2× the expected count and a fresh
  permutation each run. Fix: add `__HIP_DEVICE_COMPILE__` to the guards and give
  `is_global_store_target()` an AMDGCN implementation (`__builtin_amdgcn_is_shared` /
  `_is_private`; HIPRTC has no `__isGlobal()` and cannot assemble the PTX
  `isspacep.global` path). The two-pass scheme itself is warp-width agnostic — slots come
  from a sort over `(dest, record_ordinal)` keys, not atomic arrival order. Deterministic
  suite on MI350X: 51 failures + 7 errors → **0 failures, 92/99 passing**, including the
  bfloat16 and GPU_TO_GPU binned-reduction paths.
- **Deterministic launches cannot be graph-captured on HIP** (7 tests still gated):
  the launcher allocates temporary key/value/prefix buffers per launch, and ROCm 7.2
  permits no device allocation while a capture is active. `hipMallocAsync` on the
  dedicated non-capturing allocation stream fails with error 900 even under the relaxed
  thread capture mode, and non-pooled `hipMalloc` invalidates the capture outright (Warp
  already refuses it up front — `hip_alloc_forbidden_during_capture()` in `warp.cu`).
  The fix is to cache the temporary buffers across launches so a warm capture allocates
  nothing — the same lever as the mujoco_warp per-step scratch cache.
- **Newly surfaced**: `test_module_option_override_cuda_0` (GPU_TO_GPU scatter) fails in
  the full suite but passes when the deterministic suite runs alone — reproducible 3/3
  each way. The scatter silently drops all records (result 0.0), the signature of
  `helper.count == nullptr` at launch; previously masked by the CPU fallback. Ruled out:
  kernel-cache state (the suite passes on both a cold and a warm cache) and codegen (the
  module compiles to an identical hash either way), so it is launch-side global state
  leaking in from another test module. Pure Python launch path, so likely reproduces on
  CUDA too — check there before assuming it is a HIP issue. Bisecting which preceding
  suite triggers it is the obvious next step.
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
(4) a **compiler bug — an unreachable device `printf` costs 12x** by spilling 465 VGPRs
across the whole kernel, plus the 128-VGPR cap on kernels with no `__launch_bounds__`;
(5) **`rocprofv3` hangs or segfaults** on captured mujoco_warp runs, so there is no
kernel-level profiler for this workload; (6) minor CUDA-semantics divergences.
Original working notes below.

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
cd mujoco_warp
git apply ../warp-rocm/patches/mujoco_warp-rocm-compat.patch      # ROCm compat + scratch cache
git apply ../warp-rocm/patches/mujoco_warp-chunked-stepper.patch  # mjw.Stepper (order matters)
cd ..
# edit paths in warp-rocm/rocm-tools/slurm/*.sbatch, then:
sbatch warp-rocm/rocm-tools/slurm/warp_build.sbatch   # build + SAXPY/tile smoke test
sbatch warp-rocm/rocm-tools/slurm/mjw_probe.sbatch    # full mujoco_warp test suite
sbatch warp-rocm/rocm-tools/slurm/mjw_bench.sbatch    # benchmark suite
```

`rocm-tools/stepper_validate.py` is the Stepper's validation harness (one benchmark scene
per process; speedup vs a monolithic warm graph, plus a trajectory check against
`mjw.step` with a nondeterminism control). `STEPPER_FORCE=1` unrolls the solver loop and
requires the split, which is how the decomposition is validated on CUDA. Job templates:
`rocm-tools/slurm/stepper_{amd,nv}.sbatch`.

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
