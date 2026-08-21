"""Validate and benchmark mujoco_warp.Stepper against unmodified mjw.step.

Runs one benchmark scene per process (a failed device op can poison the HIP context, so
isolation keeps a single failure from taking down the matrix). For each scene it reports:

* the split decision and, when split, the solver iterations and host reads per step;
* ms/step for a monolithic warm-captured graph (the reference) vs the Stepper;
* max |state_stepper - state_ref| over N steps from an aligned start, alongside a control
  of two independent mjw.step runs, which bounds mujoco_warp's own run-to-run
  nondeterminism. The chunked delta is only meaningful against that control.

Usage: stepper_validate.py <assets_root> <scene_index> [chunk]
"""

import os
import sys
import time

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src.io import load_trajectory
from mujoco_warp._src.io import override_model

# STEPPER_UNROLL=1 turns off conditional graph nodes, which makes CUDA unroll the solver
# budget exactly as HIP does, so the whole auto policy (including the run-time calibration)
# can be exercised on either vendor. STEPPER_FORCE=1 also requires the split, which pins
# the decomposition itself under test and disables the calibration.
FORCE = os.environ.get("STEPPER_FORCE") == "1"
UNROLL = FORCE or os.environ.get("STEPPER_UNROLL") == "1"

# (name, mjcf, nworld, nconmax, njmax, extra put_data kwargs, model overrides, init_asleep,
#  replay npz used only to set the initial state)
SCENES = [
  ("franka_emika_panda", "scene.xml", 32768, 1, 5, {}, [], False, None),
  ("humanoid", "humanoid.xml", 8192, 24, 64, {}, [], False, None),
  ("unitree_g1_flat", "scene_flat.xml", 8192, 48, 192, {}, [], False, "shuffle_dance.npz"),
  ("three_humanoids", "three_humanoids.xml", 8192, 100, 192, {}, [], False, None),
  # SLEEP routes every solve through the compact solver and, with ntree > 1, through the
  # island mapping: the paths the prototype never touched.
  (
    "aloha_clutter", "scene_clutter.xml", 2048, 256, 384, {"nccdmax": 16, "nvmax": 56},
    ["opt.enableflags=SLEEP"], True, "pick_clutter.npz",
  ),
  ("unitree_g1_hfield", "scene_hfield.xml", 8192, 48, 192, {}, [], False, "shuffle_dance.npz"),
  ("aloha_pot", "scene_pot.xml", 8192, 24, 128, {"nccdmax": 1}, [], False, "lift_pot.npz"),
]

STATE = ("qpos", "qvel", "qacc", "act", "qfrc_constraint", "sensordata")
NSTEP = 20
NCOMPARE = 10


def _state(d):
  return {k: getattr(d, k).numpy().copy() for k in STATE}


def _delta(a, b):
  return max(float(np.abs(a[k] - b[k]).max()) if a[k].size else 0.0 for k in STATE)


def _make(mjm, mjd, nworld, nconmax, njmax, extra, overrides, init_asleep):
  m = mjw.put_model(mjm)
  if overrides:
    override_model(m, overrides)
  if UNROLL:
    m.opt.graph_conditional = False
  if init_asleep:
    mjd.tree_asleep[:] = np.arange(mjm.ntree, dtype=np.int32)
  d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax, **extra)
  return m, d


def _time(launch, nstep):
  for _ in range(3):
    launch()
  wp.synchronize_device()
  t0 = time.perf_counter()
  for _ in range(nstep):
    launch()
  wp.synchronize_device()
  return (time.perf_counter() - t0) / nstep * 1000


def run(assets_root, idx, chunk):
  name, mjcf, nworld, nconmax, njmax, extra, overrides, init_asleep, replay = SCENES[idx]
  spec = mujoco.MjSpec.from_file(f"{assets_root}/{name}/{mjcf}")
  mjm = spec.compile()
  if overrides:
    override_model(mjm, overrides)
  mjd = mujoco.MjData(mjm)
  # the benchmark scenes are only valid from their replay trajectory's initial state;
  # a plain mj_resetData leaves them interpenetrating and overflows nconmax/njmax
  if replay:
    load_trajectory(f"{assets_root}/{name}/{replay}", mjm, mjd)
  elif mjm.nkey > 0:
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
  mujoco.mj_forward(mjm, mjd)

  print(
    f"\n=== {name} nv={mjm.nv} ntree={mjm.ntree} nworld={nworld} "
    f"iterations={mjm.opt.iterations} chunk={chunk} unroll={UNROLL} force={FORCE} ===",
    flush=True,
  )

  # reference: monolithic warm-captured graph, exactly what a caller writes today
  m, d_ref = _make(mjm, mjd, nworld, nconmax, njmax, extra, overrides, init_asleep)
  for _ in range(3):
    mjw.step(m, d_ref)
  wp.synchronize_device()
  with wp.ScopedCapture() as cap:
    mjw.step(m, d_ref)
  ref_ms = _time(lambda: wp.capture_launch(cap.graph), NSTEP)

  m2, d_st = _make(mjm, mjd, nworld, nconmax, njmax, extra, overrides, init_asleep)
  st = mjw.Stepper(m2, d_st, chunk=chunk, split=True if FORCE else None)
  print(f"STEP {name} split={st.split} reason={st.reason!r}", flush=True)
  st_ms = _time(st.step, NSTEP)
  print(f"STEP {name} reference {ref_ms:8.3f} ms/step", flush=True)
  print(
    f"STEP {name} stepper   {st_ms:8.3f} ms/step  speedup={ref_ms / st_ms:5.2f}x  "
    f"iters/step={st.last_iterations} reads/step={st.last_reads} split_after={st.split}"
    f"{'' if st.split else ' (' + st.reason + ')'}",
    flush=True,
  )

  # correctness: three Datas from the same state, warmed identically. a and b are both
  # unmodified mjw.step, so a-vs-b is the nondeterminism control for c-vs-a.
  _, d_a = _make(mjm, mjd, nworld, nconmax, njmax, extra, overrides, init_asleep)
  _, d_b = _make(mjm, mjd, nworld, nconmax, njmax, extra, overrides, init_asleep)
  _, d_c = _make(mjm, mjd, nworld, nconmax, njmax, extra, overrides, init_asleep)
  for _ in range(3):
    mjw.step(m, d_a)
    mjw.step(m, d_b)
    mjw.step(m, d_c)
  wp.synchronize_device()
  # capture does not execute, so building the stepper leaves d_c's state untouched
  st_c = mjw.Stepper(m, d_c, chunk=chunk, warmup=0, split=True if FORCE else None)
  for _ in range(NCOMPARE):
    mjw.step(m, d_a)
  for _ in range(NCOMPARE):
    mjw.step(m, d_b)
  for _ in range(NCOMPARE):
    st_c.step()
  wp.synchronize_device()
  a, b, c = _state(d_a), _state(d_b), _state(d_c)
  print(
    f"STEP {name} correctness over {NCOMPARE} steps: stepper-vs-ref {_delta(c, a):.3e}  "
    f"control ref-vs-ref {_delta(a, b):.3e}  (split={st_c.split})",
    flush=True,
  )


assets_root = sys.argv[1]
idx = int(sys.argv[2])
chunk = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "auto" else None
with wp.ScopedDevice("cuda:0"):
  try:
    run(assets_root, idx, chunk)
  except Exception:
    import traceback

    print(f"STEP FAILED scene_index={idx}", flush=True)
    traceback.print_exc()
print("STEPPER_VALIDATE_DONE")
