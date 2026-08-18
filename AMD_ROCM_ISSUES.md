# ROCm graph-API findings from porting NVIDIA Warp + mujoco_warp to MI350X

*Prepared 2026-08-17 from work on [nataliakokoromyti/warp `rocm-117`](https://github.com/nataliakokoromyti/warp/tree/rocm-117).
Not yet filed — this is a staging document for a report to AMD.*

**Environment**: AMD Instinct MI350X (gfx950), ROCm 7.2.0, bare metal (not AMD's docker),
Warp 1.17.0.dev2 (AMD-Ecosystem `amd-integeration-dev` + our fixes), mujoco_warp @ ea8d067.
NVIDIA comparison numbers are from an L40S (Ada, sm_89) running **the same mujoco_warp
source** with stock `warp-lang` 1.16, measured by us on the same cluster.

Everything below is reproducible with scripts in `rocm-tools/` of the branch above.

---

## 1. Feature request (highest value): hipGraph conditional nodes

**What's missing**: CUDA 12.4+ exposes conditional graph nodes (`cudaGraphAddNode` with
`cudaGraphNodeTypeConditional`), which Warp surfaces as `wp.capture_while`. There is no HIP
equivalent in ROCm 7.2, nor on ROCm/clr `develop` as of 2026-08-17 — `hipGraphConditional*`
does not appear anywhere in the tree.

**Why it matters, quantified**: mujoco_warp's constraint solver runs "up to N iterations,
stop when converged". With conditional nodes the graph exits at convergence; without them
the full budget is unrolled into the graph and replayed every step. Two of the standard
benchmark scenes configure **100 iterations and converge in 1**:

| scene (MI350X, warm-captured graph) | configured iters | actual convergence | full budget | budget capped to 1 | **wasted** |
|---|---|---|---|---|---|
| franka_emika_panda (32768 worlds) | 100 | 1.00 | 6.479 ms/step | 1.530 ms/step | **4.24x** |
| humanoid (8192 worlds) | 100 | 1.00 | 4.767 ms/step | 1.409 ms/step | **3.38x** |
| unitree_g1_flat (8192 worlds) | 10 | 1.00 | 5.297 ms/step | 4.650 ms/step | 1.14x |

These ceilings track the measured vendor gap on the same scenes almost exactly (franka
4.37x, humanoid 3.92x behind the L40S on identical source), i.e. **conditional-node support
would close most of the remaining gap on solver-dominated workloads.**

Structural confirmation via graph node census on the same scene: the CUDA graph contains
**144 kernel nodes + 1 conditional node**; the HIP graph contains **288 unrolled kernel
nodes** and no conditional.

Repro: `rocm-tools/iter_ceiling.py`.

---

## 2. Performance bug: graph-captured allocation nodes replay ~20x more expensively than CUDA

**Symptom**: a captured graph containing `memAlloc`/`memFree` nodes (i.e. any capture taken
before the application's per-step scratch buffers exist) replays dramatically slower on
ROCm than the identical graph does on CUDA.

Same scene, same worlds, same mujoco_warp source, warm vs cold capture:

| g1_flat @8192 worlds | MI350X | L40S |
|---|---|---|
| eager (no graph) | 5.56 ms | 5.45 ms |
| **cold-captured** graph (34 memAlloc nodes) | **11.21 ms** | 3.98 ms |
| **warm-captured** graph (0 alloc nodes) | 5.09 ms | 3.79 ms |
| **cold/warm penalty** | **2.20x** | **1.06x** |

So allocation nodes cost **CUDA ~6%** and **ROCm ~120%**. Note the cold-captured graph on
ROCm is *slower than not using graphs at all*, which inverts the entire value proposition
of graph capture. The effect scales with buffer size: at 8192 worlds the per-step scratch is
hundreds of MB.

Repro: `rocm-tools/cold_vs_warm.py <scene.xml> <nworld> <nconmax> <njmax>` — prints a node
census plus replay timing for eager / cold / warm.

**Related**: blit nodes show the same pattern. A synthetic kernel-only graph replays at
~1.7 us/node while graphs mixing fill/copy nodes cost ~26 us/node
(`rocm-tools/graph_overhead.py`). We work around this in Warp by routing capture-time
`hipMemsetAsync`/`hipMemcpyAsync` through trivial kernels so captured graphs contain only
kernel nodes — a workaround that should not be necessary.

---

## 3. Correctness bug: an invalidated stream capture can never be terminated (ROCm 7.2)

**Already fixed on `develop` — this is a request to backport into a 7.2.x point release.**

Sequence (single thread, thread-local capture mode):

1. `hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal)` -> capture status 1.
2. `hipMalloc(...)` -> returns 900 and **invalidates** the capture -> status 2.
   (CUDA fails the allocation without invalidating.)
3. `hipStreamEndCapture(stream, &graph)` from **the same thread that began the capture** ->
   returns **908 `hipErrorStreamCaptureWrongThread`** and leaves status at 2.

The stream is now permanently stuck in capture state: every subsequent allocation, copy and
synchronize on it fails, and the process is unrecoverable. Documented HIP/CUDA semantics say
`hipStreamEndCapture` on an invalidated capture returns `hipErrorStreamCaptureInvalidated`
(901) *and ends the capture*, which is the intended recovery path.

**Root cause** (read from `release/rocm-rel-7.2`, `hipamd/src/hip_graph.cpp`,
`hipStreamEndCapture_common`): the invalidated branch releases the capture graph and returns
without resetting the stream's capture status; the thread-ownership entry has already been
erased earlier in the same function, so a retry takes the wrong-thread path and returns 908.

**Fix already upstream**: ROCm/clr commit `fa77aed` ("clr: Fix stream capture invalidated
state reset", 2026-05-30) adds the missing `EndCapture()` reset. Its commit message
describes exactly this symptom. As of 2026-08-17 it is on `develop` only and has not shipped
in any release — 7.2 is the newest release line, so every shipped ROCm still has the bug.

Impact for us: one deliberately-failing allocation test turned into ~3,400 cascading
failures across the Warp test suite, because the poisoned stream broke every later test in
the process.

---

## 4. Minor divergences from CUDA semantics

Lower priority, but each cost us debugging time and forced a workaround:

- **`hipStreamIsCapturing(NULL)` does not report the calling thread's capture.** CUDA's
  null-stream query reflects thread-local capture state; on ROCm it returns "none" while a
  thread-local capture is active, so a runtime cannot cheaply ask "am I inside a capture?".
  We had to fall back to "is any capture active in this process", which is conservative.
- **`hipThreadExchangeStreamCaptureMode(hipStreamCaptureModeRelaxed)` does not permit
  side-stream `hipMallocAsync`.** It returns 900 where CUDA allows the allocation. Upstream
  Warp's sort machinery relies on this to allocate scratch on a non-capturing side stream.
  (ROCm does honour relaxed mode enough not to invalidate the capture, so it is
  half-implemented rather than absent.)
- **Event nodes captured inside graphs are unreliable.** `hipEventElapsedTime` on events
  recorded inside a replayed graph returns error 400 (invalid resource handle), and
  external event record/wait nodes do not synchronize two separately-launched graphs —
  a correctness divergence: the same Warp test passes on CUDA and silently produces stale
  data on ROCm. We gate both behaviours off on HIP.
- **Device-side `printf` is lost when a kernel traps.** On an intentional device-side assert
  or out-of-bounds trap, the HSA queue aborts
  (`HSA_STATUS_ERROR_EXCEPTION ... code: 0x1016`) before buffered device printf output is
  flushed, so users see a raw hardware exception instead of Warp's diagnostic naming the
  offending array and index.

---

## What we are NOT asking for

For completeness, two things we investigated and found are *not* AMD problems:

- **The mujoco_warp cloth benchmarks' contact-buffer overflow** reproduces identically on an
  NVIDIA L40S (22/31/31 worlds vs our 26/29/29) with the same assets. Previously carried in
  our notes as an unexplained AMD-specific discrepancy; it is not.
- **Per-step scratch reallocation in mujoco_warp** (46 allocations/step, plus 7 more in the
  EPA collision path) is an upstream application issue that penalizes CUDA too; we fixed it
  application-side and contributed the fix back rather than treating it as a runtime bug.
