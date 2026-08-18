# ROCm graph-API findings from porting NVIDIA Warp + mujoco_warp to MI350X

*Prepared 2026-08-17 from work on [nataliakokoromyti/warp `rocm-117`](https://github.com/nataliakokoromyti/warp/tree/rocm-117).
Not yet filed — this is a staging document for a report to AMD.*

**Environment**: AMD Instinct MI350X (gfx950), ROCm 7.2.0, bare metal (not AMD's docker),
Warp 1.17.0.dev2 (AMD-Ecosystem `amd-integeration-dev` + our fixes), mujoco_warp @ ea8d067.
NVIDIA comparison numbers are from an L40S (Ada, sm_89) running **the same mujoco_warp
source** with stock `warp-lang` 1.16, measured by us on the same cluster.

Everything below is reproducible with scripts in `rocm-tools/` of the branch above.

---

## 0. Correctness bug (highest severity): one XCD's workgroups silently produce no output

**Under multi-process contention on one MI350X, a kernel writing to a `hipMalloc`ed buffer
can complete "successfully" while exactly one of the eight round-robin XCD classes of
workgroups leaves no visible output.** No error is reported anywhere: no launch error, no
sticky error, no HSA exception. A full `hipDeviceSynchronize` plus re-read does not repair
it -- the wrong bytes are genuinely in device memory. Memory from `hipMallocAsync` is
immune.

This was found chasing an intermittent Warp test failure and only resolved once the damage
was described structurally rather than counted. On a 1,000,000-element float32 array written
by a kernel launched with 256 threads per block:

```
runs of bad elements : 489
length of every run  : 256          <- exactly one thread block's output
stride between runs  : 2048         <- exactly eight blocks
value in every run   : 0.0          <- never written
repaired by sync     : no
```

Every corrupted-element count we have ever seen falls out of that: 489x256 = 125,184;
488x256 = 124,928; 488x256+64 = 124,992 (partial final block). Only the phase varies
between occurrences -- the missing blocks always share a single residue mod 8, and which
residue varies (we have seen 1, 2, 4, 5 and 7).

**Bisected to the allocator.** Running the same workload in 8 concurrent processes with
only the allocation path changed:

| allocation used for the buffer | processes hitting corruption |
|---|---|
| `hipMalloc` (memory pool disabled) | **2 / 8**, up to 7 hits in 1,500 iterations |
| `hipMallocAsync` (memory pool enabled) | **0 / 8** |

Controls that are clean, all at 8-way concurrency: a trivial `a[tid] = 1.0` kernel into a
pool-allocated buffer (0 in 29,000 launches), and the same with a multi-MB pageable H2D
initialization first (0 in 37,000). So it is not a host-to-device ordering problem --
**it only happens to memory that came from `hipMalloc`.** (This machine runs Compute
Partition SPX, Memory Partition NPS1.)

**Minimal reproduction** -- no copies, no graphs, no streams, no host transfers:

```python
a = hipMalloc(4 MB)                # memory pool disabled
kernel<<<3907, 256>>>(a)           # a[tid] = 1.0
hipDeviceSynchronize()
read a back                        # some blocks are still 0.0
```

Eight concurrent processes doing this hit it in **3 of 8**, roughly once per 4,000
iterations each. A representative occurrence:

```
bad_elems 124992   bad_blocks 489 of 3907   whole_blocks_missing True
block_residues_mod8 [2]   first_blocks [2, 10, 18, 26, 34, 42]
all_zero True   repaired_by_reread False
```

**Every missing block shares one residue mod 8** (2 here; 1 and 4 in other occurrences).
In SPX mode workgroups are distributed round-robin across the 8 XCDs, so this is exactly
one XCD's share of the launch producing nothing.

**The unit of loss is the thread block.** At `block_dim = 256` floats a block is exactly
1 KB, which is also the damage granularity, so we varied `block_dim` to rule out a fixed
byte lattice:

| `block_dim` | run length | run stride | elements lost (of 1,000,000) |
|---|---|---|---|
| 64 | **256 B** | **2 KB** | 124,992 |
| 256 | **1 KB** | **8 KB** | 124,928 / 125,184 |
| 1024 | **4 KB** | **32 KB** | 124,928 |

Run length is exactly `block_dim x 4` bytes and stride exactly `8 x block_dim x 4`, at every
block size, while the fraction lost stays 1/8 and the missing blocks always share a single
residue mod 8. A DMA or scrub chunking would have held a fixed byte lattice. **One XCD's
entire share of the workgroups produces no visible output.**

**Our best reading**, offered as a hypothesis rather than a measurement: a fresh
`hipMalloc` establishes a new virtual-to-physical mapping, and under multi-process
contention one XCD's TLB (or L2) is not updated for it, so that XCD's workgroups write
somewhere stale while the readback sees the correct pages still holding their scrubbed
zeros. That is consistent with everything we see: pool allocations reuse existing mappings
and are immune across 29,000+ launches under identical load; the surviving values are always
exactly `0.0`; and a full device synchronize does not repair them.

**Trigger**: process-level concurrency on a single device. One process is clean over 6,000
iterations; **eight concurrent processes doing the same work hit it in 2-7 of 8**, at
roughly one occurrence per 1,500-10,000 iterations per process. Nothing about the kernel
matters -- it reproduces with and without graph capture, on the device's stream and on a
user stream, and on whichever buffer a kernel most recently wrote.

**Impact**: a full Warp test suite run trips it in ~60% of runs, and it is a silent
wrong-answer bug, not a crash. Applications that keep memory pools enabled are not exposed,
which limits the blast radius -- but `hipMalloc` is the documented, default way to allocate
device memory, and any multi-tenant MI350X workload using it can silently read back zeros
where it wrote data.

**Repro**: `rocm-tools/block_dropout.py --no-mempool` (allocate ~4 MB with `hipMalloc`,
launch a kernel writing 1.0 to every element, synchronize, read back, look for zeros) and
`rocm-tools/copy_repro.py --mempool off`; run eight copies concurrently against one GPU.
Machine state when reproducing: Compute Partition **SPX**, Memory Partition **NPS1**,
`HSA_XNACK` and `GPU_MAX_HW_QUEUES` unset, ROCm 7.2.0 bare metal.

**Question for AMD**: is the VRAM scrub on a fresh `hipMalloc` guaranteed to be complete,
or ordered against subsequent stream work, before the pointer is returned? If so, what
breaks that guarantee under concurrent multi-process load?

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

## 4. API gap: `hipGraphAddMemFreeNode` cannot take a captured `hipMallocAsync` pointer

*Most likely a gap in our own HIP path rather than a ROCm defect -- included because the
workaround it forces is where the defect lives, and because a HIP equivalent of the CUDA
API would remove the need for a workaround at all.*

`cudaGraphAddMemFreeNode(&node, graph, deps, ndeps, ptr)` lets a runtime state exactly which
nodes a free must follow. Warp uses it to make an in-capture free depend on *every leaf node
descended from the allocation*, so the free is ordered after all uses of the allocation on
every stream in the capture. `hipGraphAddMemFreeNode` rejects a pointer obtained from
`hipMallocAsync` during stream capture (`hipErrorInvalidValue`) and only accepts pointers
from `hipGraphAddMemAllocNode` on an explicitly constructed graph, so the only available
substitute is `hipFreeAsync` on the capturing stream -- which can only express "after this
one stream's frontier". Any allocation used on a side stream inside the capture then has no
edge to its free node. Symptom below.

Warp's `test_cuda_graph_alloc_transient_stream` builds this graph on MI350X / ROCm 7.2:

1. begin capture on the device stream;
2. inside the capture, open a **temporary** side stream, allocate two 256 MB arrays from the
   memory pool, run 100 kernels over them, then let one array go out of scope so it is
   **freed inside the capture**;
3. inside the capture, open a second temporary side stream and copy a pinned host buffer
   into a third allocation;
4. end capture and launch.

Result:

```
Memory access fault by GPU node-2 (Agent handle: 0x3d950a40) on address 0xf9aee82c000.
Reason: Unknown.
```

The same test passes on CUDA — it exists specifically to catch a free that is ordered on the
wrong stream and therefore releases memory another stream is still reading. Running the
whole file one-process-per-test on an L40S with stock Warp 1.16 gives 27 pass / 0 fail /
0 crash, so every test in it is expected to hold.

What we would like from AMD: either `hipGraphAddMemFreeNode` accepting captured
`hipMallocAsync` pointers, so a runtime can name the free's dependencies the way
`cudaGraphAddMemFreeNode` allows, or a documented statement of exactly what ordering
`hipFreeAsync` on a capturing stream is guaranteed to record.

Repro: remove the `not d.is_hip` filter at the top of `warp/tests/test_graph.py` and run
`python warp/tests/test_graph.py TestGraph.test_cuda_graph_alloc_transient_stream_cuda_0`.
`rocm-tools/graph_alloc_fault.py` isolates the ingredients; measured on MI350X, the fault
requires an in-capture free **and** a side stream **and** a large buffer:

| case | result |
|---|---|
| side-stream allocation, no free | OK |
| side-stream allocation + in-capture free | **GPU fault** |
| same, `hipMalloc`-only (no fill kernels) | **GPU fault** |
| same, allocation on the capturing stream instead | OK |
| same side stream, 1/1024th the size | OK |

---

## 5. Minor divergences from CUDA semantics

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
- **An over-sized dispatch is accepted and then faults stickily.** HSA encodes each
  dimension's global work size as a uint32, so `gridDim.x * blockDim.x` must stay within
  `UINT32_MAX`. `hipModuleLaunchKernel` does not reject a larger request the way the CUDA
  driver does — it dispatches, the kernel faults asynchronously, and the launch failure is
  sticky, poisoning the context for every subsequent launch. The same applies to
  over-budget shared memory and to a `block_dim` above the function's
  `maxThreadsPerBlock`. We validate all three in our launcher because a clean synchronous
  rejection is not available. Real impact: mujoco_warp's `primitives` benchmark
  (`nworld=8192`, `nv_pad≈768` → 4.83e9 threads in one solver launch) ran on an L40S and
  could not run on MI350X until we taught the launcher to shrink the grid for grid-stride
  kernels. Repro: `rocm-tools/big_launch.py`.
- **Cross-process IPC memory does not round-trip.** `hipIpcOpenMemHandle` returns
  `hipErrorInvalidValue` for a handle exported by another process, and
  `hipIpcGetEventHandle` returns `hipErrorInvalidConfiguration` where CUDA succeeds and
  produces a handle whose invalidity is only detected on import. Warp's two IPC tests pass
  on CUDA and fail on MI350X (one with a wrong value — 84.0 where 168.0 was expected, i.e.
  the peer process's write to the shared buffer was not visible). Gated off on HIP;
  we have not investigated whether this is a configuration or a runtime limitation.

---

## What we are NOT asking for

For completeness, two things we investigated and found are *not* AMD problems:

- **The mujoco_warp cloth benchmarks' contact-buffer overflow** reproduces identically on an
  NVIDIA L40S (22/31/31 worlds vs our 26/29/29) with the same assets. Previously carried in
  our notes as an unexplained AMD-specific discrepancy; it is not.
- **Per-step scratch reallocation in mujoco_warp** (46 allocations/step, plus 7 more in the
  EPA collision path) is an upstream application issue that penalizes CUDA too; we fixed it
  application-side and contributed the fix back rather than treating it as a runtime bug.
