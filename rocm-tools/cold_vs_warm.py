"""Prove the cold-capture penalty at benchmark scale: node census + replay timing."""

import ctypes
import sys
import time
from collections import Counter

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw

xml, nworld, nconmax, njmax = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
mjm = mujoco.MjModel.from_xml_path(xml)
mjd = mujoco.MjData(mjm)
mujoco.mj_forward(mjm, mjd)

# Graph node census works on either backend: HIP and CUDA expose the same
# graph introspection entry points (hip*/cuda* names) with identical semantics.
NODE_TYPES = {0: "kernel", 1: "memcpy", 2: "memset", 3: "host", 4: "subgraph", 5: "empty", 10: "memAlloc", 11: "memFree"}
_lib, _pfx = None, None
for _name, _p in (("libamdhip64.so", "hip"), ("libcudart.so", "cuda")):
    try:
        _lib, _pfx = ctypes.CDLL(_name), _p
        break
    except OSError:
        continue


def census(graph_handle):
    if _lib is None:
        return -1, Counter()
    get_nodes = getattr(_lib, f"{_pfx}GraphGetNodes")
    get_type = getattr(_lib, f"{_pfx}GraphNodeGetType")
    n = ctypes.c_size_t(0)
    get_nodes(graph_handle, None, ctypes.byref(n))
    nodes = (ctypes.c_void_p * max(n.value, 1))()
    get_nodes(graph_handle, nodes, ctypes.byref(n))
    counts = Counter()
    for i in range(n.value):
        t = ctypes.c_int(-1)
        get_type(ctypes.c_void_p(nodes[i]), ctypes.byref(t))
        counts[NODE_TYPES.get(t.value, f"type{t.value}")] += 1
    return n.value, counts


def time_replay(graph, reps=30):
    for _ in range(5):
        wp.capture_launch(graph)
    wp.synchronize_device()
    t0 = time.perf_counter()
    for _ in range(reps):
        wp.capture_launch(graph)
    wp.synchronize_device()
    return (time.perf_counter() - t0) / reps * 1000


with wp.ScopedDevice("cuda:0"):
    m = mjw.put_model(mjm)

    # JIT all kernels first so capture doesn't include compilation
    dj = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
    for _ in range(3):
        mjw.step(m, dj)
    wp.synchronize_device()
    del dj

    # eager reference on a fresh Data
    d_eager = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
    for _ in range(3):
        mjw.step(m, d_eager)
    wp.synchronize_device()
    t0 = time.perf_counter()
    for _ in range(30):
        mjw.step(m, d_eager)
    wp.synchronize_device()
    eager_ms = (time.perf_counter() - t0) / 30 * 1000
    print(f"RESULT eager        {eager_ms:9.3f} ms/step")

    # COLD capture: fresh Data, capture on first ever step
    d_cold = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
    with wp.ScopedCapture() as cap_cold:
        mjw.step(m, d_cold)
    n_cold, c_cold = census(cap_cold.graph.graph)
    cold_ms = time_replay(cap_cold.graph)
    print(f"RESULT cold-graph   {cold_ms:9.3f} ms/step   nodes={n_cold} {dict(c_cold)}")

    # WARM capture: fresh Data, 3 eager steps first
    d_warm = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
    for _ in range(3):
        mjw.step(m, d_warm)
    wp.synchronize_device()
    with wp.ScopedCapture() as cap_warm:
        mjw.step(m, d_warm)
    n_warm, c_warm = census(cap_warm.graph.graph)
    warm_ms = time_replay(cap_warm.graph)
    print(f"RESULT warm-graph   {warm_ms:9.3f} ms/step   nodes={n_warm} {dict(c_warm)}")

    print(f"RESULT summary      cold/warm={cold_ms / warm_ms:6.2f}x  warm/eager={warm_ms / eager_ms:6.2f}x")
    print(f"RESULT throughput   warm={nworld / warm_ms * 1000:,.0f} steps/s  cold={nworld / cold_ms * 1000:,.0f} steps/s")
    print(f"RESULT nan_check    {bool(np.isnan(d_warm.qpos.numpy()).any())}")
print("COLD_VS_WARM_DONE")
