#!/usr/bin/env python3
"""Optionally strip device printf from mujoco_warp's SDF collision path.

On HIP, any device-side printf in a kernel pulls in the ockl hostcall
machinery: the compiler must reserve an implicit hostcall buffer kernarg and
treat the call as opaque, which inhibits optimization and raises register
pressure across the *whole* kernel, not just the (never-taken) error branch.

MJW_SDF_NOPRINT=1 replaces the error printfs in collision_sdf.py with no-ops so
we can measure that cost. Applied as a source rewrite because the printfs sit
inside @wp.func bodies that Warp re-parses from source.
"""

import sys

PATH = sys.argv[1]
src = open(PATH).read()

# Replace the device-side error prints inside @wp.func bodies with no-ops.
n = 0
out = []
for line in src.splitlines(keepends=True):
    stripped = line.strip()
    if stripped.startswith("wp.printf(") or stripped.startswith("wp.print("):
        indent = line[: len(line) - len(line.lstrip())]
        out.append(f"{indent}pass  # device printf removed (HIP hostcall cost experiment)\n")
        n += 1
    else:
        out.append(line)
src = "".join(out)
assert n > 0, "no device prints found"
open(PATH, "w").write(src)
print(f"stripped {n} device print(s) from {PATH}")
