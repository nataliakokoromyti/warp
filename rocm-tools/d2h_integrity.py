"""Hammer device-to-host readback and characterize any corruption.

The two intermittent Warp suite failures on MI350X are both a ``.numpy()`` that
came back with part of the buffer zeroed, at a rate around one in several
thousand copies -- far too rare for a few hundred iterations to find, which is
why the earlier ordering probes came back clean.

Observed in the archived and reproduced failures, all on 1,000,000-element
(4 MB) float32 arrays:

    124,992 / 1,000,000 mismatched, first at index 512
    124,928 / 1,000,000 mismatched, first at index 1280
    125,184 / 1,000,000 mismatched, first at index 0

Each count is ~244 x 512 elements, i.e. ~244 blocks of 2 KB, which looks like
whole chunks of a chunked transfer going missing rather than a partially
completed one. This script does many readbacks of a known pattern and, on any
mismatch, prints the *structure* of the corruption: the runs of bad indices,
their lengths, and the stride between them.

    python d2h_integrity.py --iters 20000
    python d2h_integrity.py --iters 20000 --pinned    # pinned destination
    python d2h_integrity.py --iters 20000 --load 3    # with background GPU load
"""

import argparse
import multiprocessing as mp
import time

import numpy as np

import warp as wp


@wp.kernel
def fill_ramp(a: wp.array(dtype=wp.float32), base: float):
    tid = wp.tid()
    a[tid] = base + float(tid)


def _background_load(stop_flag, device_alias):
    """Unrelated GPU work, to mimic the parallel suite runner's contention."""
    import warp as wp  # noqa: PLC0415

    wp.init()
    device = wp.get_device(device_alias)
    n = 1 << 22
    a = wp.zeros(n, dtype=wp.float32, device=device)
    b = wp.zeros(n, dtype=wp.float32, device=device)
    while not stop_flag.value:
        for _ in range(20):
            wp.copy(b, a)
        wp.synchronize_device(device)


def describe(bad_idx, n):
    """Summarize bad indices as runs, so a block pattern is visible."""
    runs = []
    start = prev = bad_idx[0]
    for i in bad_idx[1:]:
        if i != prev + 1:
            runs.append((int(start), int(prev - start + 1)))
            start = i
        prev = i
    runs.append((int(start), int(prev - start + 1)))

    lengths = sorted({length for _, length in runs})
    starts = [s for s, _ in runs]
    strides = sorted({starts[i + 1] - starts[i] for i in range(len(starts) - 1)}) if len(starts) > 1 else []
    return {
        "bad": int(bad_idx.size),
        "pct": round(100.0 * bad_idx.size / n, 2),
        "runs": len(runs),
        "run_lengths": lengths[:6],
        "first_runs": runs[:4],
        "strides": strides[:6],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--n", type=int, default=1000000)
    parser.add_argument("--pinned", action="store_true", help="read into a pinned host buffer")
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--max-reports", type=int, default=8)
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="do not re-run the producing kernel before each readback (A/B control: if the "
        "corruption is chunks of the transfer racing the producing kernel, this should be clean)",
    )
    args = parser.parse_args()

    workers = []
    stop_flag = None
    if args.load:
        ctx = mp.get_context("spawn")
        stop_flag = ctx.Value("i", 0)
        for _ in range(args.load):
            p = ctx.Process(target=_background_load, args=(stop_flag, "cuda:0"))
            p.start()
            workers.append(p)
        time.sleep(30)

    wp.init()
    device = wp.get_device("cuda:0")
    print(
        f"device: {device} n={args.n} ({args.n * 4 / 1e6:.1f} MB) iters={args.iters} "
        f"pinned={args.pinned} load={args.load}",
        flush=True,
    )

    base = 1000.0
    src = wp.zeros(args.n, dtype=wp.float32, device=device)
    wp.launch(fill_ramp, dim=args.n, inputs=[src, base], device=device)
    wp.synchronize_device(device)
    want = base + np.arange(args.n, dtype=np.float32)

    host = wp.zeros(args.n, dtype=wp.float32, device="cpu", pinned=True) if args.pinned else None

    bad_count = 0
    reports = 0
    t0 = time.perf_counter()
    for i in range(args.iters):
        # Re-run the producing kernel each iteration, with no explicit sync, so the
        # readback is issued while a write to the same buffer is still in flight. That
        # is the pattern every failing test has (interpolate/copy, then .numpy()), and
        # it is the window a chunked transfer could race into.
        if not args.no_rewrite:
            src.zero_()
            wp.launch(fill_ramp, dim=args.n, inputs=[src, base], device=device)
        if args.pinned:
            wp.copy(host, src)
            wp.synchronize_device(device)
            got = host.numpy()
        else:
            got = src.numpy()
        if not np.array_equal(got, want):
            bad_count += 1
            bad_idx = np.flatnonzero(got != want)
            if reports < args.max_reports:
                reports += 1
                zeros = int(np.count_nonzero(got[bad_idx] == 0.0))
                print(
                    f"CORRUPT iter {i}: all_bad_are_zero={zeros == bad_idx.size} {describe(bad_idx, args.n)}",
                    flush=True,
                )
            # re-read after a full sync: if the device buffer is intact, the loss was
            # in the transfer, not in the kernel that produced it
            wp.synchronize_device(device)
            again = src.numpy()
            if reports <= args.max_reports:
                print(f"        re-read after sync matches: {np.array_equal(again, want)}", flush=True)
    dt = time.perf_counter() - t0

    if stop_flag is not None:
        stop_flag.value = 1
        for p in workers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

    rate = bad_count / args.iters if args.iters else 0.0
    print(f"\n{bad_count}/{args.iters} corrupt readbacks (rate {rate:.2e}) in {dt:.1f}s", flush=True)
    return 1 if bad_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
