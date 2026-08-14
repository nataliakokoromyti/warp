"""Measure HIP graph replay overhead vs node count on this GPU."""
import time

import numpy as np
import warp as wp

wp.init()


@wp.kernel
def tiny(a: wp.array(dtype=float)):
    i = wp.tid()
    a[i] = a[i] + 1.0


a = wp.zeros(256, dtype=float, device="cuda:0")
wp.launch(tiny, dim=256, inputs=[a], device="cuda:0")
wp.synchronize()

for nodes in (10, 100, 500, 1000, 2000):
    with wp.ScopedDevice("cuda:0"):
        with wp.ScopedCapture() as cap:
            for _ in range(nodes):
                wp.launch(tiny, dim=256, inputs=[a])
        # warmup
        for _ in range(3):
            wp.capture_launch(cap.graph)
        wp.synchronize()
        reps = 20
        beg = time.perf_counter()
        for _ in range(reps):
            wp.capture_launch(cap.graph)
            wp.synchronize()
        end = time.perf_counter()
    ms = (end - beg) / reps * 1000
    print(f"nodes={nodes}: {ms:.3f} ms/launch  ({ms / nodes * 1000:.1f} us/node)")

# eager comparison at 1000 launches
beg = time.perf_counter()
for _ in range(5):
    for _ in range(1000):
        wp.launch(tiny, dim=256, inputs=[a], device="cuda:0")
    wp.synchronize()
end = time.perf_counter()
print(f"eager 1000 launches: {(end - beg) / 5 * 1000:.3f} ms")
print("GRAPH_OVERHEAD_DONE")
