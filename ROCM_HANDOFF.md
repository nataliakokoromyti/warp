# Warp + mujoco_warp on MI350X — collaborator handoff

*Goal: make Warp (and by extension mujoco_warp) fully validated and **super optimized** on
AMD Instinct MI350X (gfx950). Status as of 2026-08-17.*

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

Where the remaining gap actually lives, now that allocation noise is gone:

1. **Conditional graph nodes (still AMD-blocked).** The L40S graph has **144 kernel nodes
   plus a conditional node**; ours has **288 unrolled** kernel nodes. NVIDIA's `capture_while`
   exits the solver at convergence (~3 iterations on G1); we must replay the full fixed
   budget (10). This is structural, not tuning, and only AMD can unblock it.
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
  (`test_copy_i2c_...Graph...`, `test_implicit_fields`) — **superseded**, see "The two
  intermittent suite failures" above for the recovered signatures and the flake hunt.

## `primitives` benchmark -- root-caused and fixed (2026-08-18)

`primitives` was the last AMD-only benchmark failure (it runs on an L40S at 1.19M
steps/s). It is not a physics or rendering problem: mujoco_warp's Newton solver launches

```python
wp.launch(_update_gradient_init_h_sparse(sc), dim=(d.nworld, m.nv_pad, m.nv_pad), ...)
```

and `primitives` runs `nworld=8192` with `nv_pad` in the high hundreds --
`8192 x 768 x 768 = 4,831,838,208` threads, past `UINT32_MAX`. **HSA encodes each
dispatch dimension's global work size as a uint32**, so `gridDim.x * blockDim.x` cannot
exceed `2**32`; `wp_cuda_launch_kernel` rejected the launch up front (the guard exists
because HIP does *not* reject it -- it dispatches and faults with a sticky launch failure
that poisons the context). CUDA has no such ceiling, which is the entire vendor
difference. Verified directly: the same launch shape counts correctly on an L40S
(`rocm-tools/big_launch.py`).

**Fix** (`warp/native/warp.cu`): the blanket `dim > UINT32_MAX` rejection is only correct
for *lean* kernels, which map one thread per work item. A **grid-stride** kernel (Warp's
default) loops over the full extent, so the grid size carries no semantics and clamping
`grid_x` to `UINT32_MAX / block_dim` covers exactly the same work items. The guard now
clamps for grid-stride launches and only rejects lean ones. This also un-gates
`test_large.py`'s two `not d.is_hip` tests, which launch 2**33 and ~5.5e11 threads
through grid-stride kernels -- and those tests check exact per-work-item counts, so they
verify the clamped path covers every element.

Tool: `rocm-tools/big_launch.py` (oversized 3D and 1D launches plus a
context-still-usable check).

## The two intermittent suite failures -- characterized (2026-08-18)

Both were recovered from the archived suite logs, and they are **the same failure mode**,
not two unrelated flakes: a **torn device-to-host read**. Warp's `array.numpy()` issues its
D2H copy on the device's null stream and never explicitly synchronizes -- it relies on
CUDA's legacy null-stream ordering plus the documented rule that a D2H copy into *pageable*
host memory returns only once it has completed.

| | test | signature |
|---|---|---|
| `suite117-full-16811062` | `test_copy_i2c_d2d_SrcPoolOn_DstPoolOff_Stream0_NoGrad_Graph_AccessDstSrc` | 125,184 / 1,000,000 mismatched -- a **zero prefix**, correct suffix |
| `suite117-full-16812542` | `test_implicit_fields` | 3 / 9 mismatched -- correct prefix, **zero suffix** (`[5,5,5,5,5,5,0,0,0]`) |

The two directions are consistent with one cause. A D2H DMA walks the buffer front to
back, so a reader that starts too early on a large buffer returns a zero *prefix*; a small
buffer copied in one shot instead captures whichever elements the producing kernel had
written, giving a zero *suffix*. Neither is a lost write.

**What was tested and did not reproduce it** (all on MI350X, rocm-117; every probe is in
`rocm-tools/` and each has a clean L40S control):

| probe | result |
|---|---|
| `capture_fork_join.py` -- does a captured cross-stream fork/join (the shape `wp.copy()` builds for non-contiguous arrays) actually order? | **PASS**, 20/20. A 27 ms kernel on the forked branch makes `synchronize_stream()` block the full 27 ms, so the join edge is honored. |
| `null_stream_sync.py` -- 7 shapes of "produce on one stream, read with `.numpy()`, no explicit sync", including graph replay with no sync at all | **0 torn reads / 25 trials each**, both vendors. HIP's null-stream ordering and unpinned-D2H blocking both behave like CUDA's. |
| `copy_repro.py` -- the exact failing copy configuration, 300 iterations, plus all 32 non-contiguous d2d variants | **0 mismatches**, both vendors. |
| `flake_hunt.py --test fem_implicit` -- 500 iterations, then 300 more with 3 background GPU-load processes | **0 failures**, both vendors. |

So the mechanism is identified but the trigger is not: it needs the full parallel suite
(~16 test classes across processes sharing one GPU), which is the only context in which
either failure has ever been seen. Base rate from the archive: **2 failures in 9 recorded
full-suite runs**. `rocm-tools/slurm/flake_hunt.sbatch` repeats the whole suite N times and
dumps every failure block, which is the right next step -- a single green run does not
clear this.

**Assessment**: real, low-frequency, and *not* explained away. Do not treat a single green
suite as proof. The most likely remaining candidates, in order: (1) HIP's unpinned D2H
losing its implicit blocking under multi-process contention; (2) reuse of Warp's per-stream
`cached_event` (one event per stream, re-recorded on every `wait_stream`/`ScopedStream`
entry -- legal on CUDA, and a pattern HIP has historically been looser about).

## Test-gate audit (2026-08-18)

Every "green" suite result is only as good as what it still runs. The 240 suite skips on
MI350X break down as follows -- 140 are `add_function_test` device lists that filter HIP
out entirely (reported as *"No suitable devices to run the test"*), the rest are ordinary
capability skips shared with CUDA:

| suite | HIP-skipped | verdict |
|---|---|---|
| `deterministic/*` (4 modules) | 76 | genuine; the deterministic subsystem is unported (see below) |
| `test_graph.py` | 20 | **hides a hard GPU crash -- see below** |
| `cuda/test_texture.py` | 19 | genuine (CDNA has no texture hardware; the CPU sampling fallback is covered separately) |
| `cuda/test_cluster_dim.py` | 8 | genuine (no thread block clusters) |
| `cuda/test_clang_cuda.py` | 7 | genuine (emits PTX/CUDA that cannot load on gfx) |
| `cuda/test_streams.py` | 3 | 2 genuine HIP limitations (in-graph event timing, external event nodes), 1 timing-flaky (stream priority) |
| `test_large.py` | 2 | **fixed** -- see the `primitives` section; the grid-stride clamp lets both run |
| `test_fast_math.py` | 2 | genuine (fast-math `powf(-2,2)` divergence, PTX inspection) |
| `cuda/test_ipc.py` | 2 | gate is correct, reason was not -- see below |
| `test_bf16.py` | 1 | needs two devices |

### `test_graph.py`: the gate hides a GPU memory fault (2026-08-18)

The exclusion reads *"HIP/ROCm does not support native CUDA graph capture"*. It was
inherited from AMD's base, where capture was disabled; `rocm-117` enables capture, so the
comment is simply false and 20 tests stop running in the area this port changed most.

Removing it and running the file on MI350X:

```
test_cuda_graph_alloc_free_preserves_merged_frontier_cuda_0 ... ok
test_cuda_graph_alloc_transient_stream_cuda_0 ... Memory access fault by GPU node-2
    (Agent handle: 0x3d950a40) on address 0xf9aee82c000. Reason: Unknown.
```

`test_cuda_graph_alloc_transient_stream` allocates inside a capture on a *temporary* side
stream, lets one array go out of scope so it is freed inside the capture, and then checks
the results. Its own comment states the hazard it exists to catch:

> Array `b` goes out of scope here and is freed. If the free runs on an incorrect stream,
> the memory could be released prematurely. Other streams that are allocating memory could
> then reuse the memory while it is still used on this stream, leading to data corruption.

On MI350X it does not merely corrupt -- it faults the GPU. **This is a real bug the gate
was hiding, and it is the strongest lead for the two intermittents**: a mis-ordered
mempool free hands a still-live buffer to a later allocation, and the milder form of that
is precisely "the buffer reads back partly zero" (every async-copy test allocates its
destination from `np.zeros`, so a stale in-flight zero-fill landing on recycled memory
produces the observed zero prefix).

Where to look: `warp.cu` orders in-capture frees after their allocation with the
allocating stream's **`cached_event`** (`cuEventRecord(alloc_si->cached_event, alloc_stream)`
then `cuStreamWaitEvent(free_stream, ...)`, three sites around lines 647 / 1070 / 1229).
That is one reused event per stream; the pattern is safe under CUDA's event semantics and
is the first thing to check against HIP's.

Status: a per-test isolated sweep (`rocm-tools/isolate_tests.py`, one process per test so
the first fault does not hide the rest) enumerates which of the 20 pass, fail, or crash.
**Do not re-enable the suite until the fault is fixed** -- a crash aborts the whole test
process. But do not leave the gate labelled "capture unsupported" either; it is now
labelled as covering a known fault.

The two IPC tests were re-run with the gate removed: both genuinely fail on MI350X.
`hipIpcOpenMemHandle` returns `hipErrorInvalidValue` for a handle exported by another
process and the peer's write is not visible (84.0 read where 168.0 was expected);
`hipIpcGetEventHandle` returns `hipErrorInvalidConfiguration` where CUDA succeeds. So the
gate stays, but it was hiding a *known-broken* feature behind an "unvalidated" comment --
now stated as a measured failure in the test file and in `AMD_ROCM_ISSUES.md`.

Two other classes of gate were checked and cleared:

- **Dynamic gates** in `test_sparse.py`, `test_array.py`, `geometry/test_hash_grid.py` and
  `cuda/test_async.py` filter on `Device.supports_graph_capture`, which is now `True` on
  HIP, so they *do* run there -- only their comments still claimed otherwise. Comments
  corrected; no coverage was lost. (This is also why the async-copy `_Graph` variants run
  on HIP at all.)
- **The central skip-on-HIP hook** (`_HIP_UNSUPPORTED_ERROR_MARKERS` in
  `unittest_utils.py`) converts a matching *runtime error* into a skip. It carried two
  over-broad capture markers: `"native graph capture is unsupported"` (no raising site
  left) and `"Graph capture is not active on this stream"` (a genuine capture-state error
  that must fail loudly). Both removed; measured effect on the suite is zero (only the
  conditional-graph-node marker ever fired).

FP-tolerance relaxations were reviewed and are all narrow and justified: gfx `powf`
differs from NVIDIA's by ~1e-6 (`test_map.py` rtol 5e-6, `test_codegen.py` 4 places),
backward accumulation by ~1e-6 (`test_grad.py` tol 1e-4 on values of magnitude 10-40).
The `test_atomic_cas.py` spinlock exclusion is a correct hardware fact -- CDNA wavefronts
share an execution mask, so a GPU-wide spinlock built on `atomic_cas` deadlocks.

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
- **In-capture free on a temporary side stream faults the GPU** (found 2026-08-18):
  `test_cuda_graph_alloc_transient_stream` aborts with "Memory access fault by GPU node-2".
  The whole `test_graph.py` suite stays gated because the fault kills the test process.
  **This is the highest-priority open correctness item** -- see the gate-audit section.
- **Cross-process IPC does not work**: `hipIpcOpenMemHandle` rejects a peer handle and the
  peer's write is not visible; `hipIpcGetEventHandle` returns an error where CUDA succeeds.
  Both `test_ipc` cross-process tests gated.
- **Device-side abort loses printf output**: gfx950 HSA queue aborts (intentional traps,
  OOB asserts) fire before device printf flushes; tests accept the HSA error signature.
- **A single launch cannot exceed `UINT32_MAX` threads per dispatch dimension** (HSA
  encoding). Grid-stride kernels now clamp the grid and run correctly; lean kernels are
  rejected cleanly. See the `primitives` section.
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

### Validation gate

There is no ROCm CI runner, so validation is a single job rather than a pipeline:

```bash
sbatch rocm-tools/slurm/validate.sbatch          # build + both suites + flake watch + benchmarks
VALIDATE_QUICK=1 sbatch .../validate.sbatch      # skip benchmarks
```

It builds Warp, runs the Warp and mujoco_warp suites, repeats the historically flaky test
classes five times, smoke-tests the benchmark suite, and ends with a greppable
`### GATE <name> PASS|FAIL` block plus `OVERALL PASS|FAIL`. `WARP_DIR`/`MJW_DIR` override
which trees it validates. Run it before declaring a rebase or a port change green; the port
will otherwise rot silently as mujoco_warp main moves.

`rocm-tools/` also has the diagnostics used in this effort: `graph_overhead.py` (graph replay
cost vs node count), `graph_census.py` (node-type census of a captured step graph),
`alloc_trace.py` (which allocations fire during capture, with call sites), `g1_msgraph.py`
(single- vs multi-stream capture on G1@256; needs `hipgraph_ms.py` fetched from
zhihuidu-amd/hipgraph-ms), `sort_check.py` (segmented sort correctness), `flex_check.py`
(cloth physics vs CPU reference); and from the robustness pass: `big_launch.py`
(oversized launches past HSA's uint32 ceiling), `null_stream_sync.py` (torn `.numpy()`
reads), `capture_fork_join.py` (captured cross-stream join ordering), `copy_repro.py`
(the exact intermittent async-copy configuration), `flake_hunt.py` (repeat one test with
optional background GPU load).

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
