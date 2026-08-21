"""Decompose the in-capture allocation fault seen on MI350X.

``warp/tests/test_graph.py::test_cuda_graph_alloc_transient_stream`` aborts with
"Memory access fault by GPU node-2" on MI350X / ROCm 7.2 while passing on CUDA.
That test does several things at once, so this script splits them into cases and
runs each in its own process (a GPU fault kills the process, so cases cannot
share one).

Cases, in increasing similarity to the failing test::

    alloc_only          allocate inside the capture on a temp stream, no free
    alloc_free          ... and let one array go out of scope inside the capture
    alloc_free_nofill   ... using wp.empty, so no capture-time fill kernels run
    alloc_free_devstream ... on the device's own stream instead of a temp stream
    alloc_free_small    ... temp stream, 1/1024th the buffer size
    full                the test verbatim

Run ``python graph_alloc_fault.py`` to sweep every case, or
``python graph_alloc_fault.py <case>`` to run one.
"""

import subprocess
import sys

import numpy as np

import warp as wp

BIG = 64 * 1024 * 1024
SMALL = BIG // 1024
NUM_LAUNCHES = 100
FILL_VALUE = 42


@wp.kernel
def accum_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] + b[tid]


def _side_stream_alloc(n, free_b, fill, use_temp_stream):
    """Allocate (and optionally free) inside the capture."""
    ctx = wp.ScopedStream(wp.Stream()) if use_temp_stream else wp.ScopedStream(wp.get_device().stream)
    with ctx:
        if fill:
            a = wp.zeros(n, dtype=float)
            b = wp.ones(n, dtype=float)
        else:
            a = wp.empty(n, dtype=float)
            b = wp.empty(n, dtype=float)
        for _ in range(NUM_LAUNCHES):
            wp.launch(accum_kernel, dim=a.size, inputs=[a, b])
    if not free_b:
        return a, b
    # b goes out of scope here and is freed inside the capture
    return a, None


def run_case(case, device):
    n = SMALL if case == "alloc_free_small" else BIG
    free_b = case != "alloc_only"
    fill = case != "alloc_free_nofill"
    use_temp_stream = case != "alloc_free_devstream"

    with wp.ScopedDevice(device):
        wp.load_module(device=device)
        c_h = wp.full(n, FILL_VALUE, dtype=float, device="cpu", pinned=True)

        with wp.ScopedCapture(force_module_load=False) as capture:
            a, b = _side_stream_alloc(n, free_b, fill, use_temp_stream)
            if case == "full":
                with wp.ScopedStream(wp.Stream()):
                    c = wp.empty(n, dtype=float)
                    wp.copy(c, c_h)
            else:
                c = None

        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        if fill:
            got = a.numpy()
            want = np.full(n, NUM_LAUNCHES, dtype=np.float32)
            if not np.array_equal(got, want):
                bad = int(np.count_nonzero(got != want))
                print(f"WRONG a: {bad}/{n} mismatched, first={got[:4]} want={want[:4]}", flush=True)
                return 2
        if c is not None:
            got_c = c.numpy()
            if not np.array_equal(got_c, np.full(n, FILL_VALUE, dtype=np.float32)):
                print(f"WRONG c: first={got_c[:4]}", flush=True)
                return 2
        del b
    return 0


CASES = [
    "alloc_only",
    "alloc_free",
    "alloc_free_nofill",
    "alloc_free_devstream",
    "alloc_free_small",
    "full",
]


def main():
    if len(sys.argv) > 1:
        wp.init()
        device = wp.get_device("cuda:0")
        return run_case(sys.argv[1], device)

    # driver: one process per case
    results = {}
    for case in CASES:
        proc = subprocess.run(
            [sys.executable, __file__, case], capture_output=True, text=True, timeout=900, check=False
        )
        out = proc.stdout + proc.stderr
        if proc.returncode == 0:
            verdict = "OK"
        elif "Memory access fault" in out or "HSA_STATUS_ERROR" in out or proc.returncode < 0:
            verdict = "GPU_FAULT"
        elif proc.returncode == 2:
            verdict = "WRONG_VALUES"
        else:
            verdict = "ERROR"
        results[case] = verdict
        print(f"{verdict:12s} {case}", flush=True)
        if verdict != "OK":
            print("\n".join(out.strip().splitlines()[-12:]), flush=True)

    print(f"\nsummary: {results}", flush=True)
    return 1 if any(v != "OK" for v in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
