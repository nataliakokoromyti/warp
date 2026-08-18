"""Targeted repro for the intermittent async-copy suite failure.

Reproduces ``test_copy_i2c_d2d_SrcPoolOn_DstPoolOff_Stream0_NoGrad_Graph_AccessDstSrc``
(indexed source -> contiguous destination, same device, explicit non-default
stream, graph capture) in a tight loop, and on every mismatch reports:

* the index of the first mismatch and how many leading elements were correct
  (the reported failure signature was "correct prefix, zeros after"), and
* whether a full device synchronize followed by a re-read *heals* the array.

A self-healing mismatch proves a missing ordering edge (race), not a lost
write.  ``--variants all`` sweeps every non-contiguous d2d variant that takes
the same fork/join code path in ``wp.copy()``, plus same-stream controls.

Self-contained (no warp.tests imports) so it runs against stock Warp too.
"""

import argparse
import sys

import numpy as np

import warp as wp

N = 1000000


def as_contiguous_array(data, device=None):
    return wp.array(data=data, device=device, copy=True)


def as_strided_array(data, device=None):
    a = wp.array(data=data, device=device)
    strides = (*a.strides[:-1], 2 * a.strides[-1])
    strided_a = wp.zeros(shape=a.shape, strides=strides, dtype=a.dtype, device=device)
    wp.copy(strided_a, a)
    return strided_a


def as_indexed_array(data, device=None):
    a = wp.array(data=data, device=device)
    shape = (*a.shape[:-1], 2 * a.shape[-1])
    big_a = wp.zeros(shape=shape, dtype=a.dtype, device=device)
    indices = wp.array(data=np.arange(0, shape[-1], 2, dtype=np.int32), device=device)
    indexed_a = big_a[indices]
    wp.copy(indexed_a, a)
    return indexed_a


CTORS = {
    "contiguous": as_contiguous_array,
    "strided": as_strided_array,
    "indexed": as_indexed_array,
}


class Capturable:
    """Mirror of warp.tests.cuda.test_async.Capturable."""

    def __init__(self, use_graph=True, stream=None):
        self.use_graph = use_graph
        self.stream = stream

    def __enter__(self):
        if self.use_graph:
            wp.force_load(wp.get_device())
            wp.capture_begin(stream=self.stream, force_module_load=False)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.use_graph:
            try:
                graph = wp.capture_end(stream=self.stream)
            except Exception:
                if exc_type is None:
                    raise
            else:
                if exc_type is None:
                    wp.capture_launch(graph, stream=self.stream)


def one_copy(device, src_ctor, dst_ctor, value_offset, use_own_stream, use_graph):
    """Run one copy the way copy_template does for the d2d case."""
    src_data = np.arange(value_offset, value_offset + N, dtype=np.float32)
    dst_data = np.zeros(N, dtype=np.float32)

    with (
        wp.ScopedMempool(device, True),
        wp.ScopedMempool(device, False),
        wp.ScopedMempoolAccess(device, device, True),
    ):
        src = src_ctor(src_data, device=device)
        dst = dst_ctor(dst_data, device=device)

        if use_own_stream:
            stream = wp.Stream(device)
            stream_arg = stream
        else:
            stream = device.stream
            stream_arg = None

        wp.synchronize()
        with Capturable(use_graph=use_graph, stream=stream):
            wp.copy(dst, src, stream=stream_arg)
        wp.synchronize_stream(stream)

        got = dst.numpy()
        want = src.numpy()
        if np.array_equal(got, want):
            return None

        bad = np.flatnonzero(got != want)
        first = int(bad[0])
        n_zero_tail = int(np.count_nonzero(got[first:] == 0.0))
        wp.synchronize_device(device)
        healed = dst.numpy()
        return {
            "first_mismatch": first,
            "total_mismatch": int(bad.size),
            "tail_all_zero": bool(n_zero_tail == got.size - first),
            "healed_by_sync": bool(np.array_equal(healed, want)),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--variants", choices=["target", "all"], default="target")
    args = parser.parse_args()

    wp.init()
    device = wp.get_device("cuda:0")
    print(f"device: {device} arch={device.arch} graph_capture={getattr(device, 'supports_graph_capture', None)}", flush=True)

    if args.variants == "target":
        cases = [("indexed", "contiguous", True, True)]
    else:
        cases = []
        for s in CTORS:
            for d in CTORS:
                if s == "contiguous" and d == "contiguous":
                    continue  # pure memcpy path, no fork/join
                for own_stream in (True, False):
                    for use_graph in (True, False):
                        cases.append((s, d, own_stream, use_graph))

    failures = 0
    offset = 0
    for src_type, dst_type, own_stream, use_graph in cases:
        name = (
            f"{src_type}2{dst_type}_"
            f"{'OwnStream' if own_stream else 'DevStream'}_"
            f"{'Graph' if use_graph else 'NoGraph'}"
        )
        bad = 0
        details = []
        for _i in range(args.iters):
            offset += N
            res = one_copy(device, CTORS[src_type], CTORS[dst_type], offset, own_stream, use_graph)
            if res is not None:
                bad += 1
                if len(details) < 5:
                    details.append(res)
        status = "OK  " if bad == 0 else "FAIL"
        print(f"{status} {name}: {bad}/{args.iters} mismatches", flush=True)
        for d in details:
            print(f"       {d}", flush=True)
        failures += bad

    print(f"\ntotal mismatching iterations: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
