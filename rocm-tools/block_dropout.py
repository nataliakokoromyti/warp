"""Detect kernel launches that silently drop whole thread blocks on MI350X.

Characterizing the intermittent Warp suite failures produced this, every time,
on a 1,000,000-element float32 array written by a 256-thread-per-block kernel:

    489 runs of exactly 256 bad elements, stride 2048 elements, all zero

256 elements is exactly one thread block's output and 2048 elements is eight
blocks, so **one thread block in every eight wrote nothing**. MI350X (gfx950,
CDNA4) is an 8-XCD part and distributes workgroups round-robin across XCDs, so
"every 8th block" is "one XCD's share". A full device synchronize and re-read
does not repair it, so the writes never landed.

A trivial kernel writing into a device-allocated (``wp.zeros``) buffer does
**not** reproduce it: 0 in 29,000 launches across 8 concurrent processes. What
every failing case does instead is initialize the buffer with a **host-to-device
copy of a multi-MB numpy array** and then run the kernel over it. ``--h2d-init``
switches to that, which is the discriminating experiment: if the leftover 1 KB
blocks hold the *h2d* values rather than the kernel's, then part of a chunked
H2D transfer is landing after the kernel that was supposed to follow it.

Only reproduces under multi-process contention -- run several copies
concurrently against one GPU::

    for i in $(seq 8); do python block_dropout.py --iters 3000 --h2d-init & done; wait
"""

import argparse
import contextlib
import time

import numpy as np

import warp as wp

DEFAULT_BLOCK = 256


@wp.kernel
def write_ones(a: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    a[tid] = 1.0


def analyze(got, n, block):
    bad = np.flatnonzero(got != 1.0)
    if bad.size == 0:
        return None
    blocks = sorted(set((bad // block).tolist()))
    residues = sorted({b % 8 for b in blocks})
    whole = all(
        int(np.count_nonzero(got[b * block : (b + 1) * block] != 1.0)) in (block, n - b * block) for b in blocks
    )
    runs = []
    run_starts = []
    start = prev = int(bad[0])
    for raw in bad[1:]:
        idx = int(raw)
        if idx != prev + 1:
            runs.append(prev - start + 1)
            run_starts.append(start)
            start = idx
        prev = idx
    runs.append(prev - start + 1)
    run_starts.append(start)
    byte_stride = (run_starts[1] - run_starts[0]) * 4 if len(run_starts) > 1 else None
    return {
        "bad_elems": int(bad.size),
        "bad_blocks": len(blocks),
        "total_blocks": (n + block - 1) // block,
        # The discriminator: if the damage tracks thread blocks it scales with block_dim;
        # if it tracks a fixed byte lattice (e.g. a DMA/scrub chunking) it does not.
        "run_bytes": sorted({r * 4 for r in runs})[:4],
        "run_stride_bytes": byte_stride,
        "whole_blocks_missing": whole,
        "block_residues_mod8": residues,
        "first_blocks": blocks[:6],
        "all_zero": bool(np.all(got[bad] == 0.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--n", type=int, default=1000000)
    parser.add_argument("--max-reports", type=int, default=5)
    parser.add_argument("--block-dim", type=int, default=DEFAULT_BLOCK)
    parser.add_argument(
        "--no-mempool",
        action="store_true",
        help="allocate with the default allocator (hipMalloc) instead of the memory pool "
        "(hipMallocAsync). Bisection showed this is the trigger: the corruption appears only "
        "when the pool is disabled.",
    )
    parser.add_argument(
        "--h2d-init",
        action="store_true",
        help="initialize the buffer with a host-to-device copy of a numpy array instead of a "
        "device-side zero fill (this is what wp.array(data=...) does, and what every failing "
        "test does before the kernel that gets clobbered)",
    )
    args = parser.parse_args()

    wp.init()
    device = wp.get_device("cuda:0")
    print(
        f"device: {device} n={args.n} block_dim={args.block_dim} "
        f"blocks={(args.n + args.block_dim - 1) // args.block_dim} "
        f"iters={args.iters} h2d_init={args.h2d_init} no_mempool={args.no_mempool}",
        flush=True,
    )

    host_init = np.zeros(args.n, dtype=np.float32) if args.h2d_init else None
    pool_scope = wp.ScopedMempool(device, False) if args.no_mempool else contextlib.nullcontext()

    bad_count = 0
    reports = 0
    t0 = time.perf_counter()
    with pool_scope:
        for i in range(args.iters):
            if args.h2d_init:
                a = wp.array(data=host_init, device=device, copy=True)
            else:
                a = wp.zeros(args.n, dtype=wp.float32, device=device)
            wp.launch(write_ones, dim=args.n, inputs=[a], device=device, block_dim=args.block_dim)
            wp.synchronize_device(device)
            got = a.numpy()
            info = analyze(got, args.n, args.block_dim)
            if info is not None:
                bad_count += 1
                if reports < args.max_reports:
                    reports += 1
                    wp.synchronize_device(device)
                    info["repaired_by_reread"] = bool(np.array_equal(a.numpy(), np.ones(args.n, dtype=np.float32)))
                    print(f"DROPPED iter {i}: {info}", flush=True)
            del a
    dt = time.perf_counter() - t0

    print(f"\n{bad_count}/{args.iters} launches lost blocks in {dt:.1f}s", flush=True)
    return 1 if bad_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
