"""Log every warp array allocation that fires inside the captured G1 step."""

import traceback

import mujoco
import warp as wp

import mujoco_warp as mjw

XML = "/matx/u/knatalia/.bench_assets/unitree_g1_flat/scene_flat.xml"
mjm = mujoco.MjModel.from_xml_path(XML)
mjd = mujoco.MjData(mjm)
mujoco.mj_forward(mjm, mjd)

capturing = [False]


def hook(name, f):
    def wrapped(*a, **k):
        if capturing[0]:
            stack = traceback.extract_stack()
            caller = next(
                (fr for fr in reversed(stack[:-1]) if "alloc_trace" not in fr.filename),
                stack[-2],
            )
            print(f"ALLOC wp.{name} <- {caller.filename}:{caller.lineno} ({caller.name})", flush=True)
        return f(*a, **k)

    return wrapped


for n in ["zeros", "empty", "full", "empty_like", "zeros_like", "full_like", "ones", "clone"]:
    setattr(wp, n, hook(n, getattr(wp, n)))

with wp.ScopedDevice("cuda:0"):
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=256, nconmax=48, njmax=192)
    for _ in range(3):
        mjw.step(m, d)
    wp.synchronize()
    capturing[0] = True
    with wp.ScopedCapture() as cap:
        mjw.step(m, d)
    capturing[0] = False
print("ALLOC_TRACE_DONE")
