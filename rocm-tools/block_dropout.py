"""Detect kernel launches that silently drop whole thread blocks on MI350X.

**This is the reproducer for the MI350X data-corruption bug.** Under
multi-process contention, a kernel writing to a ``hipMalloc``ed buffer can
complete with no error while exactly one of the eight round-robin XCD classes of
workgroups leaves no visible output. A full device synchronize does not repair
it. ``hipMallocAsync`` (Warp's default memory pool) is immune.

    # reproduces: 3 of 8 processes, roughly once per 4,000 iterations each
    for i in $(seq 8); do python block_dropout.py --iters 4000 --no-mempool & done; wait

    # control: same thing with the memory pool enabled -- 0 of 8
    for i in $(seq 8); do python block_dropout.py --iters 4000 & done; wait

A representative occurrence on a 1,000,000-element float32 array, 256 threads
per block::

    bad_elems 124992   bad_blocks 489 of 3907   whole_blocks_missing True
    block_residues_mod8 [2]   first_blocks [2, 10, 18, 26, 34, 42]
    all_zero True   repaired_by_reread False

Every missing block shares one residue mod 8, and ``--block-dim`` shows the unit
of loss is the thread block rather than a fixed byte lattice: run length is
always ``block_dim * 4`` bytes and the stride ``8 * block_dim * 4``, at 64, 256
and 1024 threads per block, while the fraction lost stays 1/8.

``--h2d-init`` initializes the buffer with a host-to-device copy instead of a
device fill; it is clean (0 in 37,000 launches), which is how host-to-device
ordering was ruled out.
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
