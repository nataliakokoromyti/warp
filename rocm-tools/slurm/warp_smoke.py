import numpy as np
import warp as wp

wp.init()
d = wp.get_device("cuda:0")
print("device:", d.name, "| is_hip:", getattr(d, "is_hip", "?"))

@wp.kernel
def saxpy(x: wp.array(dtype=float), y: wp.array(dtype=float), a: float):
    i = wp.tid()
    y[i] = a * x[i] + y[i]

n = 1024
x = wp.array(np.ones(n, dtype=np.float32), device="cuda:0")
y = wp.array(np.full(n, 2.0, dtype=np.float32), device="cuda:0")
wp.launch(saxpy, dim=n, inputs=[x, y, 3.0], device="cuda:0")
wp.synchronize()
out = y.numpy()
assert np.allclose(out, 5.0), out[:8]
print("SAXPY OK:", out[:4])

@wp.kernel
def tile_sum_kernel(a: wp.array2d(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    t = wp.tile_load(a[i], shape=(64,))
    s = wp.tile_sum(t)
    wp.tile_store(out, s, offset=i)

a = wp.array(np.ones((4, 64), dtype=np.float32), device="cuda:0")
o = wp.zeros(4, dtype=float, device="cuda:0")
wp.launch_tiled(tile_sum_kernel, dim=4, inputs=[a, o], block_dim=64, device="cuda:0")
wp.synchronize()
assert np.allclose(o.numpy(), 64.0), o.numpy()
print("TILE SUM (wave64 shuffle path) OK:", o.numpy())
print("SMOKE_SUCCESS")
