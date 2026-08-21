"""Probe captured cross-stream fork/join ordering.

``wp.copy()`` between non-contiguous arrays forks the capture onto the
destination device's stream and joins back:

    dest.device.stream.wait_stream(stream)   # fork
    wp_array_copy_device(...)                # kernel on dest.device.stream
    stream.wait_stream(dest.device.stream)   # join

If the captured join edge is honored, launching the graph on ``stream`` and
then calling ``wp.synchronize_stream(stream)`` must block until the forked
kernel finishes.  This script measures exactly that: a deliberately slow
kernel is placed on the forked branch, and we time the host sync.

    join honored  -> host sync takes ~= kernel duration
    join dropped  -> host sync returns immediately and the destination reads
                     back partially written

Run with no arguments; prints a PASS/FAIL verdict per configuration.
"""

import argparse
import time

import numpy as np

import warp as wp

N = 1 << 20


@wp.kernel
def slow_write(a: wp.array(dtype=wp.float32), spin: int):
    tid = wp.tid()
    acc = float(0.0)
    for _i in range(spin):
        acc = acc * 1.0000001 + 1.0
    a[tid] = acc


def time_ms(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1e3


def build_graph(device, stream, a, spin, fork):
    """Capture on `stream`; optionally fork the slow kernel onto device.stream."""
    wp.load_module(device=device)
    wp.capture_begin(stream=stream, force_module_load=False)
    try:
        if fork:
            device.stream.wait_stream(stream)
            wp.launch(slow_write, dim=N, inputs=[a, spin], stream=device.stream)
            stream.wait_stream(device.stream)
        else:
            wp.launch(slow_write, dim=N, inputs=[a, spin], stream=stream)
    finally:
        graph = wp.capture_end(stream=stream)
    return graph


def run_case(device, spin, fork, trials):
    stream = wp.Stream(device)
    a = wp.zeros(N, dtype=wp.float32, device=device)
    graph = build_graph(device, stream, a, spin, fork)

    # warm up (first replay pays instantiation costs)
    wp.capture_launch(graph, stream=stream)
    wp.synchronize_device(device)

    sync_ms = []
    zeros_after = []
    for _ in range(trials):
        a.zero_()
        wp.synchronize_device(device)

        def _launch_and_sync():
            wp.capture_launch(graph, stream=stream)
            wp.synchronize_stream(stream)

        sync_ms.append(time_ms(_launch_and_sync))

        # read back exactly the way the test suite does (dst.numpy())
        host = a.numpy()
        zeros_after.append(int(np.count_nonzero(host == 0.0)))

    return {
        "sync_ms_mean": float(np.mean(sync_ms)),
        "sync_ms_min": float(np.min(sync_ms)),
        "sync_ms_max": float(np.max(sync_ms)),
        "zero_elems_max": int(np.max(zeros_after)),
        "trials_with_zeros": int(sum(1 for z in zeros_after if z > 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spin", type=int, default=20000)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    wp.init()
    device = wp.get_device("cuda:0")
    print(f"device: {device} arch={device.arch} graph_capture={getattr(device, 'supports_graph_capture', None)}")

    # Reference: how long does the slow kernel actually take, eagerly?
    a = wp.zeros(N, dtype=wp.float32, device=device)
    wp.launch(slow_write, dim=N, inputs=[a, args.spin], device=device)
    wp.synchronize_device(device)
    eager_ms = time_ms(
        lambda: (wp.launch(slow_write, dim=N, inputs=[a, args.spin], device=device), wp.synchronize_device(device))
    )
    print(f"eager kernel duration: {eager_ms:.2f} ms (spin={args.spin})")

    control = run_case(device, args.spin, fork=False, trials=args.trials)
    print(f"CONTROL  (no fork, kernel on capture stream): {control}")

    forked = run_case(device, args.spin, fork=True, trials=args.trials)
    print(f"FORKJOIN (kernel on device.stream branch)   : {forked}")

    # Verdict: the forked sync must take about as long as the control sync.
    threshold = 0.5 * control["sync_ms_mean"]
    ok = forked["sync_ms_mean"] >= threshold and forked["trials_with_zeros"] == 0
    print(f"\ncontrol sync {control['sync_ms_mean']:.2f} ms, forked sync {forked['sync_ms_mean']:.2f} ms")
    print(f"VERDICT: {'PASS - captured fork/join is honored' if ok else 'FAIL - captured join edge NOT honored'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
