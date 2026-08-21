"""Check that launches beyond HSA's uint32 global-work-size ceiling work on HIP.

HSA encodes each dispatch dimension's global work size as a uint32, so
``gridDim.x * blockDim.x`` cannot exceed ``UINT32_MAX``.  A grid-stride kernel
loops over the full extent, so clamping the grid is semantics-preserving and
such a launch should still run and cover every work item.  A lean
(``grid_stride=False``) kernel maps one thread per work item and is spread
across a 3D grid instead.

This is the shape that ``mujoco_warp``'s ``_update_gradient_init_h_sparse``
launch takes on the ``primitives`` benchmark: ``dim=(nworld, nv_pad, nv_pad)``
with ``nworld=8192`` and ``nv_pad`` in the high hundreds exceeds 2**32 threads.
"""

import warp as wp


@wp.kernel(grid_stride=True)
def count_first_row(dim_y: int, result: wp.array(dtype=wp.uint64)):
    i, j, k = wp.tid()
    if j == 0 and k == 0:
        wp.atomic_add(result, 0, wp.uint64(1))


@wp.kernel(grid_stride=True)
def count_all(result: wp.array(dtype=wp.uint64)):
    wp.atomic_add(result, 0, wp.uint64(1))


def main():
    wp.init()
    device = wp.get_device("cuda:0")
    print(
        f"device: {device} arch_str={getattr(device, 'arch_str', device.arch)} is_hip={getattr(device, 'is_hip', False)}",
        flush=True,
    )

    result = wp.zeros(1, dtype=wp.uint64, device=device)

    # mujoco_warp primitives shape: 8192 * 768 * 768 = 4,831,838,208 > 2**32
    nworld, nv_pad = 8192, 768
    total = nworld * nv_pad * nv_pad
    print(f"launching dim=({nworld}, {nv_pad}, {nv_pad}) = {total} threads (2**32 = {2**32})", flush=True)
    wp.launch(count_first_row, dim=(nworld, nv_pad, nv_pad), inputs=[nv_pad, result], device=device)
    wp.synchronize_device(device)
    got = int(result.numpy()[0])
    ok3d = got == nworld
    print(f"3D oversized launch: counted {got}, expected {nworld} -> {'OK' if ok3d else 'FAIL'}", flush=True)

    # 1D just past the ceiling
    result.zero_()
    dim1d = 2**32 + 12345
    print(f"launching dim={dim1d} threads", flush=True)
    wp.launch(count_all, dim=dim1d, inputs=[result], device=device)
    wp.synchronize_device(device)
    got1d = int(result.numpy()[0])
    ok1d = got1d == dim1d
    print(f"1D oversized launch: counted {got1d}, expected {dim1d} -> {'OK' if ok1d else 'FAIL'}", flush=True)

    # A subsequent normal launch must still work (an over-sized dispatch used to
    # poison the context with a sticky launch failure).
    result.zero_()
    wp.launch(count_all, dim=1000, inputs=[result], device=device)
    wp.synchronize_device(device)
    got_after = int(result.numpy()[0])
    ok_after = got_after == 1000
    print(f"context still usable: counted {got_after}, expected 1000 -> {'OK' if ok_after else 'FAIL'}", flush=True)

    return 0 if (ok3d and ok1d and ok_after) else 1


if __name__ == "__main__":
    raise SystemExit(main())
