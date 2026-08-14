"""Enumerate node types in a captured mujoco_warp step graph via HIP APIs."""

import ctypes
from collections import Counter

import mujoco
import warp as wp

import mujoco_warp as mjw

XML = "/matx/u/knatalia/.bench_assets/unitree_g1_flat/scene_flat.xml"
mjm = mujoco.MjModel.from_xml_path(XML)
mjd = mujoco.MjData(mjm)
mujoco.mj_forward(mjm, mjd)

hip = ctypes.CDLL("libamdhip64.so")

NODE_TYPES = {
    0: "kernel",
    1: "memcpy",
    2: "memset",
    3: "host",
    4: "subgraph",
    5: "empty",
    6: "waitEvent",
    7: "eventRecord",
    8: "extSemSignal",
    9: "extSemWait",
    10: "memAlloc",
    11: "memFree",
}


def census(graph_handle, label):
    n = ctypes.c_size_t(0)
    hip.hipGraphGetNodes(graph_handle, None, ctypes.byref(n))
    nodes = (ctypes.c_void_p * max(n.value, 1))()
    hip.hipGraphGetNodes(graph_handle, nodes, ctypes.byref(n))
    counts = Counter()
    for i in range(n.value):
        t = ctypes.c_int(-1)
        hip.hipGraphNodeGetType(ctypes.c_void_p(nodes[i]), ctypes.byref(t))
        counts[t.value] += 1
    print(f"{label}: {n.value} nodes")
    for t, c in sorted(counts.items()):
        print(f"  type {t:2d} ({NODE_TYPES.get(t, 'unknown')}): {c}")


with wp.ScopedDevice("cuda:0"):
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=256, nconmax=48, njmax=192)
    for _ in range(3):
        mjw.step(m, d)
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        mjw.step(m, d)
    assert cap.graph.graph, "graph handle not retained"
    census(cap.graph.graph, "warm-captured step graph")
print("CENSUS_DONE")
