"""Probe legacy NULL-stream implicit synchronization.

``warp.array.numpy()`` issues its device-to-host copy on the device's *null*
stream and relies on CUDA's legacy null-stream semantics for ordering::

    with warp.ScopedStream(self.device.null_stream):
        a = self.to("cpu", requires_grad=False)

Legacy semantics say work on the NULL stream is implicitly ordered after all
pending work on every *blocking* stream of the device (Warp creates its streams
with ``CU_STREAM_DEFAULT``, i.e. blocking).  There is no explicit synchronize
anywhere on this path.

If that implicit ordering is not honored, ``.numpy()`` races the producing
kernel and returns a torn read.  Because a D2H DMA walks the buffer front to
back, a reader that starts too early yields a **zero prefix with a correct
suffix**; a small buffer copied instantly instead yields a **correct prefix
with a zero suffix**.  Both signatures were observed in the Warp suite on
MI350X (12.5% zero prefix in an async-copy test; 3 zeros of 9 in an FEM test).

Each case writes a nonzero value into a zeroed array with a deliberately slow
kernel and then reads it back with no explicit synchronization.  Any nonzero
count means the implicit ordering was violated.
"""

import argparse

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


def zeros_report(host):
    z = np.flatnonzero(host == 0.0)
    if z.size == 0:
        return None
    return {
        "zeros": int(z.size),
        "first_zero": int(z[0]),
        "last_zero": int(z[-1]),
        "prefix_zero": bool(z[0] == 0),
    }


def case_eager_device_stream(device, a, spin):
    """Launch on device.stream (blocking), read via null stream."""
    wp.launch(slow_write, dim=N, inputs=[a, spin], device=device)
    return a.numpy()


def case_eager_user_stream(device, a, spin):
    """Launch on a separate blocking stream, read via null stream."""
    stream = wp.Stream(device)
    wp.launch(slow_write, dim=N, inputs=[a, spin], stream=stream)
    return a.numpy()


def _build_graph(device, stream, a, spin, fork):
    wp.force_load(device)
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


def make_graph_case(device, a, spin, fork, sync_stream):
    stream = wp.Stream(device)
    graph = _build_graph(device, stream, a, spin, fork)
    wp.capture_launch(graph, stream=stream)
    wp.synchronize_device(device)

    def run(device, a, spin):
        wp.capture_launch(graph, stream=stream)
        if sync_stream:
            wp.synchronize_stream(stream)
        return a.numpy()

    return run


def case_control(device, a, spin):
    """Explicit device synchronize before the read -- must always be clean."""
    wp.launch(slow_write, dim=N, inputs=[a, spin], device=device)
    wp.synchronize_device(device)
    return a.numpy()


def stress(device, sizes, spin, iters):
    """Hammer the plain launch-then-.numpy() pattern across buffer sizes.

    The FEM failure was a 9-element read (correct prefix, zero suffix) and the
    async-copy failure a 1M-element read (zero prefix, correct suffix), so the
    race may only open at particular transfer sizes.
    """
    print(f"\nstress: {iters} iterations per size, spin={spin}", flush=True)
    bad_total = 0
    for n in sizes:
        a = wp.zeros(n, dtype=wp.float32, device=device)
        wp.launch(slow_write, dim=n, inputs=[a, spin], device=device)
        wp.synchronize_device(device)
        bad = 0
        sample = None
        for _ in range(iters):
            a.zero_()
            wp.synchronize_device(device)
            wp.launch(slow_write, dim=n, inputs=[a, spin], device=device)
            rep = zeros_report(a.numpy())
            if rep is not None:
                bad += 1
                sample = sample or rep
            wp.synchronize_device(device)
        status = "OK  " if bad == 0 else "TORN"
        print(f"{status} stress_n={n}: {bad}/{iters} torn reads {sample or ''}", flush=True)
        bad_total += bad
    return bad_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spin", type=int, default=200000)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--stress-iters", type=int, default=0, help="extra per-size stress iterations")
    args = parser.parse_args()

    wp.init()
    device = wp.get_device("cuda:0")
    print(f"device: {device} arch_str={getattr(device, 'arch_str', device.arch)} is_hip={getattr(device, 'is_hip', False)}", flush=True)

    a = wp.zeros(N, dtype=wp.float32, device=device)
    # warm up / compile
    wp.launch(slow_write, dim=N, inputs=[a, args.spin], device=device)
    wp.synchronize_device(device)

    cases = {
        "control_explicit_sync": case_control,
        "eager_device_stream": case_eager_device_stream,
        "eager_user_stream": case_eager_user_stream,
        "graph_user_stream_synced": make_graph_case(device, a, args.spin, fork=False, sync_stream=True),
        "graph_user_stream_unsynced": make_graph_case(device, a, args.spin, fork=False, sync_stream=False),
        "graph_forkjoin_synced": make_graph_case(device, a, args.spin, fork=True, sync_stream=True),
        "graph_forkjoin_unsynced": make_graph_case(device, a, args.spin, fork=True, sync_stream=False),
    }

    bad_total = 0
    for name, fn in cases.items():
        bad = 0
        samples = []
        for _ in range(args.trials):
            a.zero_()
            wp.synchronize_device(device)
            host = fn(device, a, args.spin)
            rep = zeros_report(host)
            if rep is not None:
                bad += 1
                if len(samples) < 3:
                    samples.append(rep)
            wp.synchronize_device(device)
        status = "OK  " if bad == 0 else "TORN"
        print(f"{status} {name}: {bad}/{args.trials} torn reads", flush=True)
        for s in samples:
            print(f"       {s}", flush=True)
        bad_total += bad

    if args.stress_iters:
        bad_total += stress(device, [9, 1024, 1 << 16, 1 << 20], args.spin // 20, args.stress_iters)

    print(f"\ntotal torn reads: {bad_total}")
    return 1 if bad_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
