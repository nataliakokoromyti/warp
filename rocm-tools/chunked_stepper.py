"""Prototype: emulate conditional-graph early exit with host-side chunked solver graphs.

HIP has no conditional graph nodes, so mujoco_warp unrolls the full solver budget into the
captured graph. The franka/humanoid benchmark scenes budget 100 iterations and converge in
1, so ~99 iterations of dead work replay every step (measured ceiling: 4.24x / 3.38x).

This splits the step into three captured graphs -- pre (forward + solver init), chunk (K
solver iterations), post (solver tail + sensor_acc + integrator) -- and drives them from
the host, reading `nsolving` between chunks to stop at convergence. Cost: one small D2H
read per chunk (~tens of us) against ~99 skipped iterations.

Correctness is checked against unmodified mjw.step from an identical initial state.
"""

import sys
import time

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src import forward as fwd
from mujoco_warp._src import sensor as sens
from mujoco_warp._src import solver as slv
from mujoco_warp._src import types as mjt


def _forward_no_solver_iters(m, d):
    """forward() with the solver's iteration loop removed (init + tail still run)."""
    fwd.fwd_position(m, d, factorize=False)
    d.sensordata.zero_()
    sens.sensor_pos(m, d)
    fwd._energy_pos(m, d)
    fwd.fwd_velocity(m, d)
    sens.sensor_vel(m, d)
    fwd._energy_vel(m, d)
    if not (m.opt.disableflags & mjt.DisableBit.ACTUATION):
        if m.callback.control:
            m.callback.control(m, d)
    fwd.fwd_actuation(m, d)
    fwd.fwd_acceleration(m, d, factorize=True)
    # solver: init only (iterations=0 makes _solve skip its loop)
    saved = m.opt.iterations
    m.opt.iterations = 0
    try:
        slv.solve(m, d)
    finally:
        m.opt.iterations = saved


def _integrate(m, d):
    if m.opt.integrator == mjt.IntegratorType.EULER:
        fwd.euler(m, d)
    elif m.opt.integrator == mjt.IntegratorType.RK4:
        fwd.rungekutta4(m, d)
    else:
        fwd.implicit(m, d)


class ChunkedStepper:
    def __init__(self, m, d, chunk=1, warmup=3):
        self.m, self.d, self.chunk = m, d, chunk
        self.max_chunks = max(1, (m.opt.iterations + chunk - 1) // chunk)
        for _ in range(warmup):
            mjw.step(m, d)
        wp.synchronize_device()
        if "solver.nsolving" not in getattr(d, "_scratch_arrays", {}):
            raise RuntimeError(
                "ChunkedStepper requires a warmed Data: step it at least once before "
                "constructing with warmup=0, otherwise the convergence counter is "
                "allocated inside the capture and cannot be read from the host."
            )
        ctx = slv._cached_solver_context(m, d)
        self.ctx = ctx
        with wp.ScopedCapture() as cap:
            _forward_no_solver_iters(m, d)
        self.g_pre = cap.graph
        nsolving = d._scratch_arrays["solver.nsolving"]
        self.nsolving = nsolving
        with wp.ScopedCapture() as cap:
            for _ in range(chunk):
                slv._solver_iteration(m, d, ctx, nsolving)
        self.g_chunk = cap.graph
        with wp.ScopedCapture() as cap:
            sens.sensor_acc(m, d)
            _integrate(m, d)
        self.g_post = cap.graph

    def step(self):
        wp.capture_launch(self.g_pre)
        for _ in range(self.max_chunks):
            wp.capture_launch(self.g_chunk)
            if int(self.nsolving.numpy()[0]) == 0:  # implicit sync
                break
        wp.capture_launch(self.g_post)


def run(xml, nworld, nconmax, njmax, chunk, nsteps=20):
    mjm = mujoco.MjModel.from_xml_path(xml)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    name = xml.split("/")[-2]
    print(f"\n=== {name} nv={mjm.nv} nworld={nworld} iterations={mjm.opt.iterations} chunk={chunk} ===", flush=True)

    with wp.ScopedDevice("cuda:0"):
        m = mjw.put_model(mjm)

        # reference: monolithic warm-captured graph
        d_ref = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
        for _ in range(3):
            mjw.step(m, d_ref)
        wp.synchronize_device()
        with wp.ScopedCapture() as cap:
            mjw.step(m, d_ref)
        g_ref = cap.graph
        for _ in range(3):
            wp.capture_launch(g_ref)
        wp.synchronize_device()
        t0 = time.perf_counter()
        for _ in range(nsteps):
            wp.capture_launch(g_ref)
        wp.synchronize_device()
        ref_ms = (time.perf_counter() - t0) / nsteps * 1000
        ref_qpos = d_ref.qpos.numpy().copy()

        # chunked stepper from the same starting state
        d_ck = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
        st = ChunkedStepper(m, d_ck, chunk=chunk)
        for _ in range(3):
            st.step()
        wp.synchronize_device()
        t0 = time.perf_counter()
        for _ in range(nsteps):
            st.step()
        wp.synchronize_device()
        ck_ms = (time.perf_counter() - t0) / nsteps * 1000

        print(f"CHUNK reference (monolithic warm graph) {ref_ms:8.3f} ms/step", flush=True)
        print(f"CHUNK chunked stepper (chunk={chunk})    {ck_ms:8.3f} ms/step   speedup={ref_ms / ck_ms:5.2f}x", flush=True)

        # correctness: same number of steps from identical state, compare trajectories
        d_a = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
        d_b = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
        # identical warmup on both so the trajectories start aligned; capture does not
        # execute, so building the stepper leaves d_b's state untouched
        for _ in range(3):
            mjw.step(m, d_a)
            mjw.step(m, d_b)
        wp.synchronize_device()
        st_b = ChunkedStepper(m, d_b, chunk=chunk, warmup=0)
        for _ in range(10):
            mjw.step(m, d_a)
        wp.synchronize_device()
        for _ in range(10):
            st_b.step()
        wp.synchronize_device()
        a, b = d_a.qpos.numpy(), d_b.qpos.numpy()
        err = float(np.abs(a - b).max())
        print(f"CHUNK correctness: max|qpos_ref - qpos_chunked| = {err:.3e} over 10 steps", flush=True)
    return ref_ms, ck_ms, err


SCENES = [
    ("franka_emika_panda/scene.xml", 32768, 1, 5),
    ("humanoid/humanoid.xml", 8192, 24, 64),
    ("unitree_g1_flat/scene_flat.xml", 8192, 48, 192),
]

A = sys.argv[1]
# one scene per process: a failed device op can poison the HIP context, so isolation
# keeps a single failure from taking down the rest of the matrix
idx = int(sys.argv[2]) if len(sys.argv) > 2 else None
chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 1
scene, nworld, nconmax, njmax = SCENES[idx]
try:
    run(f"{A}/{scene}", nworld, nconmax, njmax, chunk)
except Exception:
    import traceback

    print(f"CHUNK FAILED {scene} chunk={chunk}", flush=True)
    traceback.print_exc()
print("CHUNKED_STEPPER_DONE")
