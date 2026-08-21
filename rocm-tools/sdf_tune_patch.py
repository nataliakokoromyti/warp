#!/usr/bin/env python3
"""Patch mujoco_warp's collision_sdf.py to make the narrowphase launch tunable.

Env knobs (read at import time):
  MJW_SDF_BLOCK_DIM   block size for _sdf_narrowphase (0 = stock: default 256, no launch_bounds)
  MJW_SDF_MIN_BLOCKS  second __launch_bounds__ arg (HIP: min waves per execution unit)
"""

import sys

PATH = sys.argv[1]
src = open(PATH).read()

anchor = 'wp.set_module_options({"enable_backward": False, "default_grid_stride": False})'
assert anchor in src, "anchor 1 missing"
src = src.replace(
    anchor,
    anchor
    + """

# --- ROCm collision tuning (agent-collision experiment) ----------------------
# _sdf_narrowphase inlines the whole gradient-descent / line-search / octree
# stack into a single kernel. Without __launch_bounds__ the HIP compiler must
# assume a 1024-thread workgroup (16 waves/CU => 4 waves/SIMD) and caps the
# kernel at 128 VGPRs, forcing scratch spills. Declaring the true block size
# lifts that cap. MJW_SDF_BLOCK_DIM=0 restores stock behaviour (the control).
_SDF_BLOCK_DIM = int(_os.environ.get("MJW_SDF_BLOCK_DIM", "0"))
_SDF_MIN_BLOCKS = int(_os.environ.get("MJW_SDF_MIN_BLOCKS", "1"))
_SDF_LAUNCH_BOUNDS = (_SDF_BLOCK_DIM, _SDF_MIN_BLOCKS) if _SDF_BLOCK_DIM > 0 else None
_SDF_LAUNCH_BLOCK = _SDF_BLOCK_DIM if _SDF_BLOCK_DIM > 0 else 256
""",
)
src = src.replace("import warp as wp", "import os as _os\n\nimport warp as wp", 1)

assert "@wp.kernel\ndef _sdf_narrowphase(" in src, "anchor 2 missing"
src = src.replace(
    "@wp.kernel\ndef _sdf_narrowphase(",
    "@wp.kernel(launch_bounds=_SDF_LAUNCH_BOUNDS)\ndef _sdf_narrowphase(",
    1,
)

tail = src.rstrip()
assert tail.endswith("  )"), repr(tail[-40:])
src = tail[: -len("  )")] + "    block_dim=_SDF_LAUNCH_BLOCK,\n  )\n"

open(PATH, "w").write(src)
print("patched", PATH)
