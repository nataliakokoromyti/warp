"""Verify wp.utils.segmented_sort_pairs correctness on this GPU."""
import numpy as np
import warp as wp

wp.init()
rng = np.random.default_rng(0)
for nseg, seglen in [(8, 578), (32, 1024), (8, 4096), (128, 100)]:
    n = nseg * seglen
    keys_np = rng.random(n).astype(np.float32)
    vals_np = np.arange(n, dtype=np.int32)
    keys = wp.zeros(2 * n, dtype=float, device="cuda:0")
    vals = wp.zeros(2 * n, dtype=int, device="cuda:0")
    wp.copy(keys, wp.array(keys_np, dtype=float, device="cuda:0"), count=n)
    wp.copy(vals, wp.array(vals_np, dtype=int, device="cuda:0"), count=n)
    starts = wp.array(np.arange(nseg + 1) * seglen, dtype=wp.int32, device="cuda:0")
    wp.utils.segmented_sort_pairs(keys, vals, n, starts)
    wp.synchronize()
    k = keys.numpy()[:n]
    v = vals.numpy()[:n]
    ok = True
    for s in range(nseg):
        seg = k[s * seglen : (s + 1) * seglen]
        if not np.all(np.diff(seg) >= 0):
            ok = False
            bad = int(np.argmin(np.diff(seg) >= 0))
            print(f"  seg {s}: NOT SORTED at {bad}: {seg[bad]} > {seg[bad + 1]}")
            break
        vseg = v[s * seglen : (s + 1) * seglen]
        if not np.allclose(np.sort(keys_np[s * seglen : (s + 1) * seglen]), seg) or not np.all(
            keys_np[vseg] == seg
        ):
            ok = False
            print(f"  seg {s}: PAIRS MISMATCHED")
            break
    print(f"nseg={nseg} seglen={seglen}: {'OK' if ok else 'BROKEN'}")
print("SORT_CHECK_DONE")
