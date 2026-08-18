#!/usr/bin/env python3
# Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Put mujoco_warp's SDF device error prints behind a compile-time debug flag.

This is the upstream-shaped version of the "strip the printfs" experiment. The
prints sit on unreachable error paths, but on HIP each one lowers to an OCKL
hostcall that the compiler must treat as opaque and memory-clobbering, so it
forces live values across every call site; six of them inside the octree walk
cost 12x on gfx950. Guarding them with a module-level Python constant means Warp
emits ``if (false)`` and the branch dies in the first simplification pass, while
the diagnostics stay one environment variable away.

Usage: python sdf_debugprint_patch.py <collision_sdf.py>
"""

import sys

PATH = sys.argv[1]
src = open(PATH).read()

anchor = 'wp.set_module_options({"enable_backward": False, "default_grid_stride": False})'
assert anchor in src, "module-options anchor missing"
src = src.replace(
    anchor,
    anchor
    + """

# Device-side error diagnostics below are on unreachable paths, but they are not
# free: on HIP every wp.printf lowers to an OCKL hostcall, which the compiler
# must treat as an opaque memory-clobbering call and spill live values across.
# Six of them inside the octree walk drive _sdf_narrowphase's register demand and
# cost 12x on gfx950. Compile them out by default; MJW_SDF_DEBUG_PRINT=1 brings
# them back.
_SDF_DEBUG_PRINT = bool(int(_os.environ.get("MJW_SDF_DEBUG_PRINT", "0")))
""",
)
src = src.replace("import warp as wp", "import os as _os\n\nimport warp as wp", 1)

n = 0
out = []
for line in src.splitlines(keepends=True):
    stripped = line.strip()
    if stripped.startswith("wp.printf(") or stripped.startswith("wp.print("):
        indent = line[: len(line) - len(line.lstrip())]
        out.append(f"{indent}if _SDF_DEBUG_PRINT:\n")
        out.append("  " + line)
        n += 1
    else:
        out.append(line)
assert n > 0, "no device prints found"
open(PATH, "w").write("".join(out))
print(f"guarded {n} device print(s) in {PATH}")
