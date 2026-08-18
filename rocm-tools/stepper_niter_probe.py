"""Why does the stepper's trajectory drift from mjw.step on pendula but not on franka?

Runs the same 20 steps with the reference, a control, and steppers pinned to several
batch sizes, and prints per-key deltas plus the solver iteration distribution.
"""

import dataclasses

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw
from mujoco_warp import test_data

STATE = ("qpos", "qvel", "qacc", "act", "qfrc_constraint", "sensordata")
NSTEP = 20


def state(d):
  return {k: getattr(d, k).numpy().copy() for k in STATE}


def delta(a, b):
  return {k: (float(np.abs(a[k] - b[k]).max()) if a[k].size else 0.0) for k in STATE}


def main(path):
  mjm, _, _, _ = test_data.fixture(path, nworld=1)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  m = dataclasses.replace(m, opt=dataclasses.replace(m.opt, graph_conditional=False))
  print(f"\n### {path} nv={mjm.nv} iterations={mjm.opt.iterations} solver={mjm.opt.solver} cone={mjm.opt.cone}")

  datas = {k: mjw.put_data(mjm, mjd, nworld=8) for k in ("ref", "ctl", "c1", "c4", "cbig")}
  for _ in range(3):
    for d in datas.values():
      mjw.step(m, d)
  wp.synchronize_device()

  steppers = {
    "c1": mjw.Stepper(m, datas["c1"], chunk=1, warmup=0, split=True),
    "c4": mjw.Stepper(m, datas["c4"], chunk=4, warmup=0, split=True),
    "cbig": mjw.Stepper(m, datas["cbig"], chunk=int(m.opt.iterations), warmup=0, split=True),
  }
  for _ in range(NSTEP):
    mjw.step(m, datas["ref"])
    mjw.step(m, datas["ctl"])
    for k, st in steppers.items():
      st.step()
  wp.synchronize_device()

  ref = state(datas["ref"])
  print("solver_niter ref :", np.unique(datas["ref"].solver_niter.numpy()))
  for k in ("ctl", "c1", "c4", "cbig"):
    d = delta(state(datas[k]), ref)
    extra = f"  iters={steppers[k].last_iterations}" if k in steppers else "  (control)"
    print(f"{k:5s} " + "  ".join(f"{n}={v:.3e}" for n, v in d.items()) + extra)
    print(f"      solver_niter {np.unique(datas[k].solver_niter.numpy())}")


with wp.ScopedDevice("cuda:0"):
  for p in ("pendula.xml", "humanoid/humanoid.xml", "constraints.xml"):
    try:
      main(p)
    except Exception:
      import traceback

      traceback.print_exc()
print("PROBE_DONE")
