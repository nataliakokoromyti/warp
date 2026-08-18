# Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Report register and scratch usage for the kernels in a Warp kernel cache.

Reads the AMDGPU metadata note out of each cached ``.cubin`` (an HSA code object
on ROCm) and prints the numbers that decide whether a kernel spills:
``vgpr_count`` against ``max_flat_workgroup_size``, and ``vgpr_spill_count`` /
``private_segment_fixed_size``. A kernel with no ``__launch_bounds__`` reports
``max_flat_workgroup_size: 1024``, which caps it at 128 VGPRs -- anything the
kernel needs beyond that shows up as spill.

Usage:
  python hsaco_regs.py <kernel-cache-dir-or-cubin> [--filter SUBSTR]

Requires ``llvm-readelf`` (any recent LLVM, or the one in ``/opt/rocm/llvm/bin``).
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys

FIELDS = (
    "max_flat_workgroup_size",
    "vgpr_count",
    "agpr_count",
    "sgpr_count",
    "vgpr_spill_count",
    "sgpr_spill_count",
    "private_segment_fixed_size",
    "group_segment_fixed_size",
)


def _readelf() -> str:
    for cand in ("llvm-readelf", "/opt/rocm/llvm/bin/llvm-readelf", "readelf"):
        if shutil.which(cand) or pathlib.Path(cand).exists():
            return cand
    raise SystemExit("no llvm-readelf found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="kernel cache directory or a single .cubin/.hsaco")
    ap.add_argument("--filter", default="", help="only report objects whose path contains this")
    args = ap.parse_args()

    root = pathlib.Path(args.path)
    objs = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix in (".cubin", ".hsaco"))
    objs = [p for p in objs if args.filter in str(p)]
    if not objs:
        print("no code objects found")
        return 1

    tool = _readelf()
    width = max(len(f) for f in FIELDS)
    for obj in objs:
        try:
            out = subprocess.run([tool, "--notes", str(obj)], capture_output=True, text=True, errors="replace").stdout
        except OSError as e:
            print(f"{obj}: {e}")
            continue
        vals = {}
        for f in FIELDS:
            m = re.search(rf"{f}:\s*(\d+)", out)
            if m:
                vals[f] = int(m.group(1))
        if not vals:
            continue
        print(f"=== {obj}")
        for f in FIELDS:
            if f in vals:
                flag = ""
                if f == "vgpr_spill_count" and vals[f]:
                    flag = "  <-- SPILLING"
                if f == "max_flat_workgroup_size" and vals[f] == 1024:
                    flag = "  <-- no __launch_bounds__ (VGPRs capped at 128)"
                print(f"  {f:<{width}} {vals[f]:>10}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
