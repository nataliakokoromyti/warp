"""Find allocations that still fire during a WARM capture (should be zero)."""

import sys
import traceback

import mujoco
import warp as wp

import mujoco_warp as mjw

xml, nworld, nconmax, njmax = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
mjm = mujoco.MjModel.from_xml_path(xml)
mjd = mujoco.MjData(mjm)
mujoco.mj_forward(mjm, mjd)

tracing = [False]
hits = []


def hook(name, f):
    def wrapped(*a, **k):
        if tracing[0]:
            stack = traceback.extract_stack()
            caller = next((fr for fr in reversed(stack[:-1]) if "warm_alloc_trace" not in fr.filename), stack[-2])
            shape = a[0] if a else k.get("shape") or k.get("n")
            hits.append(f"wp.{name}{'':1s} shape={shape} <- {caller.filename.split('/')[-1]}:{caller.lineno} ({caller.name})")
        return f(*a, **k)

    return wrapped


for n in ["zeros", "empty", "full", "empty_like", "zeros_like", "full_like", "ones", "clone"]:
    setattr(wp, n, hook(n, getattr(wp, n)))

with wp.ScopedDevice("cuda:0"):
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
    for _ in range(5):  # generous warmup
        mjw.step(m, d)
    wp.synchronize_device()
    tracing[0] = True
    with wp.ScopedCapture() as cap:
        mjw.step(m, d)
    tracing[0] = False

print(f"WARMALLOC scene={xml.split('/')[-2]} allocations_during_warm_capture={len(hits)}")
for h in sorted(set(hits)):
    print(f"WARMALLOC   {h}  (x{hits.count(h)})")
print("WARM_ALLOC_TRACE_DONE")
