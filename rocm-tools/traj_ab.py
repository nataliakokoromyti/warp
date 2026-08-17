"""Eager vs cold-graph vs warm-graph on a real ctrl trajectory (testspeed-faithful)."""

import sys
import time

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src import cli

xml, npz, nworld, nconmax, njmax, nstep = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    int(sys.argv[4]),
    int(sys.argv[5]),
    int(sys.argv[6]),
)
mjm = mujoco.MjModel.from_xml_path(xml)
mjd = mujoco.MjData(mjm)
mujoco.mj_forward(mjm, mjd)
ctrls = cli.load_trajectory(npz, mjm, mjd)
nstep = min(nstep, len(ctrls))


def fresh_data():
    return mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)


def set_ctrl(d, i):
    d.ctrl.assign(np.broadcast_to(ctrls[i], (nworld, len(ctrls[i]))).copy())


with wp.ScopedDevice("cuda:0"):
    m = mjw.put_model(mjm)

    # JIT everything once on a throwaway Data
    d0 = fresh_data()
    for _ in range(3):
        mjw.step(m, d0)
    wp.synchronize_device()
    del d0

    def run(mode):
        d = fresh_data()
        graph = None
        if mode == "cold-graph":
            with wp.ScopedCapture() as cap:
                mjw.step(m, d)
            graph = cap.graph
        elif mode == "warm-graph":
            for _ in range(3):
                mjw.step(m, d)
            wp.synchronize_device()
            with wp.ScopedCapture() as cap:
                mjw.step(m, d)
            graph = cap.graph
        total = 0.0
        for i in range(nstep):
            set_ctrl(d, i)
            wp.synchronize_device()
            t0 = time.perf_counter()
            if graph is None:
                mjw.step(m, d)
            else:
                wp.capture_launch(graph)
            wp.synchronize_device()
            total += time.perf_counter() - t0
        nan = bool(np.isnan(d.qpos.numpy()).any())
        print(f"MODE {mode:11s} {total / nstep * 1000:8.3f} ms/step  ({nworld * nstep / total:12,.0f} steps/s) nan={nan}")

    run("eager")
    run("cold-graph")
    run("warm-graph")
print("TRAJ_AB_DONE")
