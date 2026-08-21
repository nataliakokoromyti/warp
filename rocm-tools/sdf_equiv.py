# Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dump mujoco_warp SDF narrowphase contact output for an A/B equivalence check.

Steps a scene to a fixed state and writes the contact arrays the SDF narrowphase
produced to an ``.npz``. Run it once per source variant and diff the files: that
is what proves a codegen or source change did not move the physics.

It also leaves the process's stdout untouched, so a device-side ``wp.printf``
firing from inside the kernel (mujoco_warp's SDF octree walk has several error
prints) is visible in the job log rather than swallowed by a grep filter.

Usage:
  python sdf_equiv.py <scene.xml> --nworld N --out a.npz [--nconmax N] [--njmax N]
  python sdf_equiv.py --compare a.npz b.npz
"""

import argparse
import sys

import numpy as np


def compare(path_a: str, path_b: str) -> int:
    a, b = np.load(path_a), np.load(path_b)
    worst = 0.0
    for k in sorted(a.files):
        if k not in b.files:
            print(f"{k}: missing in {path_b}")
            return 1
        xa, xb = a[k], b[k]
        if xa.shape != xb.shape:
            print(f"{k}: shape {xa.shape} vs {xb.shape}")
            return 1
        if xa.dtype.kind == "f":
            d = float(np.max(np.abs(xa - xb))) if xa.size else 0.0
        else:
            d = float(np.max(np.abs(xa.astype(np.int64) - xb.astype(np.int64)))) if xa.size else 0.0
        worst = max(worst, d)
        print(f"  {k:<20} max_abs_diff={d:.6e}")
    print(f"EQUIV worst_max_abs_diff={worst:.6e}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", nargs="?")
    ap.add_argument("--nworld", type=int, default=512)
    ap.add_argument("--nconmax", type=int, default=None)
    ap.add_argument("--njmax", type=int, default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    if not args.xml or not args.out:
        ap.error("need <scene.xml> and --out, or --compare a.npz b.npz")

    import mujoco
    import mujoco_warp as mjw

    import warp as wp

    mjm = mujoco.MjModel.from_xml_path(args.xml)
    mjd = mujoco.MjData(mjm)
    if mjm.nkey > 0:
        mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    kw = {k: getattr(args, k) for k in ("nconmax", "njmax") if getattr(args, k) is not None}
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=args.nworld, **kw)

    for _ in range(args.steps):
        mjw.step(m, d)
    wp.synchronize_device()

    nacon = int(d.nacon.numpy()[0])
    n = min(nacon, d.naconmax)
    print(f"# nacon={nacon} naconmax={d.naconmax}")
    np.savez(
        args.out,
        nacon=np.array([nacon]),
        dist=d.contact.dist.numpy()[:n],
        pos=d.contact.pos.numpy()[:n],
        frame=d.contact.frame.numpy()[:n],
        geom=d.contact.geom.numpy()[:n],
        worldid=d.contact.worldid.numpy()[:n],
        qpos=d.qpos.numpy(),
        qvel=d.qvel.numpy(),
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
