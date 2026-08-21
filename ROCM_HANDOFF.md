# Warp + mujoco_warp on MI350X — collaborator handoff

*Goal: make Warp (and by extension mujoco_warp) fully validated and **super optimized** on
AMD Instinct MI350X (gfx950). Status as of 2026-08-18.*

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
**recovered in software 2026-08-18 by `mjw.Stepper`**, see below; **(2) collision kernels** —
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
   `mjw.Stepper` (section above): host-driven solver batches buy 3.86x on franka, 3.07x on
   humanoid, 1.72x on three_humanoids, at the cost of ~1 host read per step. Still worth
   asking AMD for real conditional nodes -- they would remove the reads entirely and the
   ~1-2% they cost on scenes that use their whole iteration budget -- but this is no longer
   a blocker.
2. **Collision-dominated scenes.** `aloha_sdf` (12.2x) and `unitree_g1_hfield` (10.4x)
   barely improved from warm capture -- their cost is collision compute, not allocation.
   These are the top targets for our own optimization work.
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
