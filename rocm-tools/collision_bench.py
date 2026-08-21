# Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Time mujoco_warp's collision phase in isolation, on either vendor.

End-to-end ``testspeed`` numbers mix collision compute with the solver, whose
cost on HIP is inflated by the missing conditional-graph early exit. This tool
steps the model into a representative state, then repeatedly runs *only* the
collision pipeline, and isolates each narrowphase by differential timing (run
the pipeline with that narrowphase stubbed out and subtract).

Usage:
  python collision_bench.py <scene.xml> --nworld N [--nconmax N] [--njmax N] ...
"""

import argparse
import statistics
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
from mujoco_warp._src import collision_driver
from mujoco_warp._src import io as mjw_io

import warp as wp


def _time(fn, reps, inner):
    """Median and min wall-ms per call of fn."""
    fn()
    wp.synchronize_device()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        wp.synchronize_device()
        samples.append((time.perf_counter() - t0) * 1e3 / inner)
    return statistics.median(samples), min(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("--nworld", type=int, default=8192)
    ap.add_argument("--nconmax", type=int, default=None)
    ap.add_argument("--njmax", type=int, default=None)
    ap.add_argument("--nccdmax", type=int, default=None)
    ap.add_argument("--nvmax", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--label", default="x")
    ap.add_argument("--keyframe", type=int, default=0)
    ap.add_argument("--init_asleep", action="store_true")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--replay", default=None, help="NPZ ctrl trajectory; replayed to --replay_step before timing")
    ap.add_argument("--replay_step", type=int, default=200, help="how far into the trajectory to advance")
    args = ap.parse_args()

    mjm = mujoco.MjModel.from_xml_path(args.xml)
    mjd = mujoco.MjData(mjm)
    # Mirror mjwarp-testspeed: reset to a keyframe, do NOT mj_forward (which would
    # populate mjd.contact and trip put_data's nconmax check).
    ctrls = None
    if args.replay:
        ctrls = mjw_io.load_trajectory(args.replay, mjm, mjd)
    elif mjm.nkey > 0 and args.keyframe > -1:
        mujoco.mj_resetDataKeyframe(mjm, mjd, args.keyframe)

    put_kwargs = {k: getattr(args, k) for k in ("nconmax", "njmax", "nccdmax", "nvmax") if getattr(args, k) is not None}
    m = mjw.put_model(mjm)
    if args.override:
        mjw_io.override_model(m, args.override)
    if args.init_asleep:
        mjd.tree_asleep[:] = np.arange(mjm.ntree, dtype=np.int32)
    d = mjw.put_data(mjm, mjd, nworld=args.nworld, **put_kwargs)

    # Advance into a representative state. With a replay trajectory this matches
    # the state the benchmark suite actually measures (e.g. hfield's robot walking
    # over terrain generates far more contacts than the settled keyframe pose).
    if ctrls is not None:
        nrep = min(args.replay_step, len(ctrls))
        for i in range(nrep):
            d.ctrl.assign(np.tile(ctrls[i], (args.nworld, 1)).astype(np.float32))
            mjw.step(m, d)
    else:
        for _ in range(args.warmup):
            mjw.step(m, d)
    wp.synchronize_device()

    dev = wp.get_device()
    print(f"# device={dev} scene={args.xml} nworld={args.nworld} label={args.label}")
    print(f"# nacon={int(d.nacon.numpy()[0])} naconmax={d.naconmax} has_sdf={m.has_sdf_geom} nflex={m.nflex}")

    out = {}
    out["step"] = _time(lambda: mjw.step(m, d), args.reps, args.inner)
    out["collision"] = _time(lambda: collision_driver.collision(m, d), args.reps, args.inner)

    # differential isolation of each narrowphase
    noop = lambda *a, **k: None  # noqa: E731
    for name in ("sdf_narrowphase", "convex_narrowphase", "primitive_narrowphase"):
        orig = getattr(collision_driver, name)
        setattr(collision_driver, name, noop)
        try:
            out[f"collision_minus_{name}"] = _time(lambda: collision_driver.collision(m, d), args.reps, args.inner)
        finally:
            setattr(collision_driver, name, orig)

    # broadphase only: stub all three narrowphases
    saved = {
        n: getattr(collision_driver, n) for n in ("sdf_narrowphase", "convex_narrowphase", "primitive_narrowphase")
    }
    for n in saved:
        setattr(collision_driver, n, noop)
    try:
        out["broadphase_only"] = _time(lambda: collision_driver.collision(m, d), args.reps, args.inner)
    finally:
        for n, f in saved.items():
            setattr(collision_driver, n, f)

    for k, (med, mn) in out.items():
        print(f"RESULT {args.label} {k} median_ms={med:.4f} min_ms={mn:.4f}")

    col = out["collision"][0]
    for name in ("sdf_narrowphase", "convex_narrowphase", "primitive_narrowphase"):
        print(f"RESULT {args.label} isolated_{name} ms={col - out[f'collision_minus_{name}'][0]:.4f}")
    print(f"RESULT {args.label} isolated_broadphase ms={out['broadphase_only'][0]:.4f}")
    print(f"RESULT {args.label} collision_frac={col / out['step'][0]:.4f}")


if __name__ == "__main__":
    main()
