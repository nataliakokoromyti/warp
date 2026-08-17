"""How much is solver early-exit worth?

NVIDIA's captured graph puts the solver loop behind a conditional node and exits at
convergence (measured niter_mean: franka 1.0, humanoid 1.4, G1 3.0). HIP has no
conditional nodes, so we replay the full fixed iteration budget. Capping
m.opt.iterations bounds what a working early-exit would buy us.
"""

import sys
import time

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw

A = sys.argv[1]
SCENES = [
    ("franka_emika_panda/scene.xml", 32768, 1, 5),
    ("humanoid/humanoid.xml", 8192, 24, 64),
    ("unitree_g1_flat/scene_flat.xml", 8192, 48, 192),
]


def bench(mjm, mjd, nworld, nconmax, njmax, iters, mode):
    m = mjw.put_model(mjm)
    if iters is not None:
        m.opt.iterations = iters
    d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
    for _ in range(3):
        mjw.step(m, d)
    wp.synchronize_device()
    graph = None
    if mode == "warm-graph":
        with wp.ScopedCapture() as cap:
            mjw.step(m, d)
        graph = cap.graph
        for _ in range(5):
            wp.capture_launch(graph)
        wp.synchronize_device()
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        if graph is None:
            mjw.step(m, d)
        else:
            wp.capture_launch(graph)
    wp.synchronize_device()
    ms = (time.perf_counter() - t0) / reps * 1000
    niter = float(d.solver_niter.numpy().mean())
    nan = bool(np.isnan(d.qpos.numpy()).any())
    return ms, niter, nan


with wp.ScopedDevice("cuda:0"):
    for scene, nworld, nconmax, njmax in SCENES:
        mjm = mujoco.MjModel.from_xml_path(f"{A}/{scene}")
        mjd = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, mjd)
        print(f"\n=== {scene.split('/')[0]} nv={mjm.nv} nworld={nworld} default_iters={mjm.opt.iterations} ===", flush=True)
        base = None
        for iters in (None, 4, 2, 1):
            for mode in ("warm-graph",):
                try:
                    ms, niter, nan = bench(mjm, mjd, nworld, nconmax, njmax, iters, mode)
                    if base is None:
                        base = ms
                    label = f"iterations={'default' if iters is None else iters}"
                    print(
                        f"ITER {label:22s} {mode:11s} {ms:8.3f} ms  niter_mean={niter:5.2f} "
                        f"speedup={base / ms:5.2f}x nan={nan}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"ITER {iters} {mode} FAILED: {type(e).__name__} {str(e)[:120]}", flush=True)
print("ITER_CEILING_DONE")
