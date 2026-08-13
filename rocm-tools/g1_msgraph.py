"""G1 256 worlds: multi-stream (GlobalMode) graph capture vs single-stream vs eager."""
import time

import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mjw
import hipgraph_ms

XML = "/matx/u/knatalia/.bench_assets/unitree_g1_flat/scene_flat.xml"
mjm = mujoco.MjModel.from_xml_path(XML)
mjd = mujoco.MjData(mjm)
mujoco.mj_forward(mjm, mjd)


def bench_graph(graph, label, d):
    for _ in range(5):
        wp.capture_launch(graph)
    wp.synchronize()
    reps = 100
    beg = time.perf_counter()
    for _ in range(reps):
        wp.capture_launch(graph)
    wp.synchronize()
    end = time.perf_counter()
    ms = (end - beg) / reps * 1000
    nan = bool(np.isnan(d.qpos.numpy()).any())
    print(f"{label}: {ms:.2f} ms/step  {256.0 / ((end - beg) / reps):,.0f} world-steps/s  nan={nan}")


with wp.ScopedDevice("cuda:0"):
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=256, nconmax=48, njmax=192)

    with wp.ScopedCapture() as cap:
        mjw.step(m, d)
    bench_graph(cap.graph, "single-stream (wp.ScopedCapture)", d)

    m2 = mjw.put_model(mjm)
    d2 = mjw.put_data(mjm, mjd, nworld=256, nconmax=48, njmax=192)
    with hipgraph_ms.MultiStreamCapture(wp.get_device("cuda:0")) as mcap:
        mjw.step(m2, d2)
    bench_graph(mcap.graph, "multi-stream (GlobalMode)", d2)

print("G1_MS_DONE")
