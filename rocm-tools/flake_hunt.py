"""Repeat individual Warp tests many times to characterize intermittent failures.

The full-suite runner executes test classes in parallel processes sharing one
GPU, so a flake may only appear under contention.  This harness repeats a
chosen test in-process N times and can spawn background GPU load to recreate
that contention.

Examples::

    python flake_hunt.py --test fem_implicit --iters 500
    python flake_hunt.py --test fem_implicit --iters 500 --load 4
    python flake_hunt.py --test async_copy --iters 200 --load 4
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
import unittest

import warp as wp


def _background_load(stop_flag, device_alias):
    """Keep the GPU busy with unrelated work (mimics parallel suite processes)."""
    import numpy as np  # noqa: PLC0415

    import warp as wp  # noqa: PLC0415

    wp.init()
    device = wp.get_device(device_alias)
    n = 1 << 22
    a = wp.array(np.random.rand(n).astype(np.float32), device=device)
    b = wp.zeros(n, dtype=wp.float32, device=device)
    while not stop_flag.value:
        for _ in range(20):
            wp.copy(b, a)
        wp.synchronize_device(device)


class _Harness(unittest.TestCase):
    def runTest(self):
        pass


def get_test(name, device):
    """Return a zero-arg callable running one iteration of the named test."""
    harness = _Harness()

    if name == "fem_implicit":
        from warp.tests.fem.test_fem_field import test_implicit_fields  # noqa: PLC0415

        return lambda: test_implicit_fields(harness, device)

    if name == "fem_field_all":
        import warp.tests.fem.test_fem_field as m  # noqa: PLC0415

        fns = [getattr(m, f) for f in dir(m) if f.startswith("test_") and callable(getattr(m, f))]

        def run_all():
            for fn in fns:
                fn(harness, device)

        return run_all

    if name == "async_copy":
        from warp.tests.cuda.test_async import (  # noqa: PLC0415
            CopyParams,
            as_contiguous_array,
            as_indexed_array,
            copy_template,
        )

        state = {"offset": 0}

        def run_copy():
            state["offset"] += 1000000
            params = CopyParams(
                src_use_mempool=True,
                dst_use_mempool=False,
                access_dst_src=True,
                access_src_dst=False,
                stream_device=device,
                with_grad=False,
                use_graph=True,
                value_offset=state["offset"],
            )
            copy_template(harness, as_indexed_array, as_contiguous_array, device, device, 1000000, params)

        return run_copy

    raise SystemExit(f"unknown test '{name}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load", type=int, default=0, help="background GPU load processes")
    args = parser.parse_args()

    workers = []
    stop_flag = None
    if args.load:
        ctx = mp.get_context("spawn")
        stop_flag = ctx.Value("i", 0)
        for i in range(args.load):
            env_cache = os.environ.get("XDG_CACHE_HOME", "")
            os.environ["XDG_CACHE_HOME"] = f"{env_cache}-load{i}" if env_cache else ""
            p = ctx.Process(target=_background_load, args=(stop_flag, args.device))
            p.start()
            workers.append(p)
            if env_cache:
                os.environ["XDG_CACHE_HOME"] = env_cache
        time.sleep(30)  # let the load processes JIT and start hammering

    wp.init()
    device = wp.get_device(args.device)
    print(f"device: {device} arch={device.arch} load_procs={args.load}", flush=True)

    run = get_test(args.test, device)

    failures = []
    t0 = time.perf_counter()
    for i in range(args.iters):
        try:
            run()
        except Exception as e:
            failures.append((i, repr(e)))
            print(f"--- ITER {i} FAILED ---", flush=True)
            traceback.print_exc()
            if len(failures) >= 20:
                print("too many failures; stopping early", flush=True)
                break
    dt = time.perf_counter() - t0

    if stop_flag is not None:
        stop_flag.value = 1
        for p in workers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

    print(f"\n{args.test}: {len(failures)} failures in {i + 1} iterations ({dt:.1f}s)", flush=True)
    for idx, msg in failures:
        print(f"  iter {idx}: {msg[:400]}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
