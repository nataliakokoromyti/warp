"""Sweep mujoco_warp BlockDim knobs on gfx950 (wave64).

Hypothesis: mujoco_warp's block sizes are NVIDIA-tuned. put_model derives the CG
solver widths as clamp(round_up_32(nv), 32, 256) and several static defaults are
32 -- on a 64-lane wavefront both waste half of every wave, and warp's tile ops
(including the rocWMMA MFMA path, which needs a full wavefront) cannot engage
below 64. Overrides are applied in-process so the library stays untouched.
"""

import sys
import time

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw

SUB_WAVE = [  # static defaults of 32
    "actuator_velocity",
    "energy_vel_kinetic",
    "cholesky_factorize",
    "cholesky_factorize_solve",
    "update_gradient_cholesky_blocked",
    "linesearch_iterative",
    "contact_jac_tiled",
    "qderiv_actuator_dense",
]
NV_DERIVED = [  # put_model sets these to round_up_32(nv)
    "update_gradient_grad",
    "solve_beta_accumulate",
    "solve_search_update_cg",
    "solve_init_search_cg",
]
HOT = [
    "cholesky_factorize_solve",
    "linesearch_iterative",
    "update_gradient_cholesky_blocked",
    "update_gradient_grad",
    "contact_jac_tiled",
    "cholesky_factorize",
]


def run_scene(xml, nworld, nconmax, njmax, reps=20):
    mjm = mujoco.MjModel.from_xml_path(xml)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    nv = mjm.nv
    blk32 = max(32, min(256, ((nv + 31) // 32) * 32))
    blk64 = max(64, min(256, ((nv + 63) // 64) * 64))
    print(f"\n=== {xml.split('/')[-2]} nv={nv} nworld={nworld} | nv_block: 32-round={blk32} 64-round={blk64} ===", flush=True)

    def measure(overrides, label):
        m = mjw.put_model(mjm)
        for k, v in overrides.items():
            setattr(m.block_dim, k, v)
        d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
        for _ in range(3):
            mjw.step(m, d)
        wp.synchronize_device()
        t0 = time.perf_counter()
        for _ in range(reps):
            mjw.step(m, d)
        wp.synchronize_device()
        ms = (time.perf_counter() - t0) / reps * 1000
        nan = bool(np.isnan(d.qpos.numpy()).any())
        flag = ""
        if base[0] is not None:
            delta = (1 - ms / base[0]) * 100
            flag = f"  {delta:+6.1f}%" + ("  <-- WIN" if delta > 2 else "")
        print(f"TUNE {label:48s} {ms:8.3f} ms/step nan={nan}{flag}", flush=True)
        return ms

    base = [None]
    base[0] = measure({}, "baseline (nvidia-tuned)")
    wave = {k: 64 for k in SUB_WAVE}
    wave.update({k: blk64 for k in NV_DERIVED})
    measure(wave, f"WAVE-AWARE (sub-wave->64, nv-derived->{blk64})")
    measure({k: 64 for k in SUB_WAVE}, "sub-wavefront knobs -> 64 only")
    measure({k: blk64 for k in NV_DERIVED}, f"nv-derived -> {blk64} only")
    for knob in HOT:
        for val in (64, 128):
            measure({knob: val}, f"{knob}={val}")


# sizing mirrors the benchmark suite invocations exactly
scenes = [
    (sys.argv[1] + "/franka_emika_panda/scene.xml", 32768, 1, 5),
    (sys.argv[1] + "/humanoid/humanoid.xml", 8192, 24, 64),
    (sys.argv[1] + "/unitree_g1_flat/scene_flat.xml", 8192, 48, 192),
]
with wp.ScopedDevice("cuda:0"):
    for xml, nworld, nconmax, njmax in scenes:
        try:
            run_scene(xml, nworld, nconmax, njmax)
        except Exception as e:
            print(f"SCENE FAILED {xml}: {type(e).__name__} {str(e)[:200]}", flush=True)
print("BLOCKDIM_TUNE_DONE")
