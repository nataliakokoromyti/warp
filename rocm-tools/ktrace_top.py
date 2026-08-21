# Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Aggregate a rocprofv3 --kernel-trace CSV into a per-kernel time ranking.

Usage:
  python ktrace_top.py <kernel_trace.csv> [--top N] [--tail-frac F]

``--tail-frac`` keeps only the last fraction of dispatches (by start time) so
warmup/JIT dispatches do not pollute the steady-state ranking.
"""

import argparse
import csv
import sys
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--tail-frac", type=float, default=0.5)
    args = ap.parse_args()

    rows = []
    with open(args.csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        print("no rows")
        return 1

    def col(row, *names):
        for n in names:
            if n in row:
                return row[n]
        raise KeyError(names)

    recs = []
    for r in rows:
        start = int(col(r, "Start_Timestamp", "start_timestamp"))
        end = int(col(r, "End_Timestamp", "end_timestamp"))
        name = col(r, "Kernel_Name", "kernel_name")
        grid = col(r, "Grid_Size", "grid_size") if ("Grid_Size" in r or "grid_size" in r) else ""
        wg = col(r, "Workgroup_Size", "workgroup_size") if ("Workgroup_Size" in r or "workgroup_size" in r) else ""
        vgpr = ""
        for k in ("VGPR_Count", "Accum_VGPR_Count", "vgpr_count"):
            if k in r:
                vgpr = r[k]
                break
        sgpr = r.get("SGPR_Count", r.get("sgpr_count", ""))
        scratch = r.get("Private_Segment_Size", r.get("private_segment_size", ""))
        lds = r.get("Group_Segment_Size", r.get("group_segment_size", ""))
        recs.append((start, end - start, name, grid, wg, vgpr, sgpr, scratch, lds))

    recs.sort()
    keep = recs[int(len(recs) * (1.0 - args.tail_frac)) :]
    span = keep[-1][0] - keep[0][0]

    agg = defaultdict(lambda: [0, 0])  # name -> [total_ns, count]
    meta = {}
    for _, dur, name, grid, wg, vgpr, sgpr, scratch, lds in keep:
        a = agg[name]
        a[0] += dur
        a[1] += 1
        meta.setdefault(name, (grid, wg, vgpr, sgpr, scratch, lds))

    total = sum(v[0] for v in agg.values())
    print(f"dispatches={len(keep)} (of {len(recs)})  busy={total / 1e6:.3f} ms  wall_span={span / 1e6:.3f} ms")
    print(f"{'%busy':>7} {'ms':>10} {'n':>7} {'us/call':>9} {'vgpr':>5} {'sgpr':>5} {'scr':>6} {'lds':>6}  kernel")
    for name, (ns, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[: args.top]:
        grid, wg, vgpr, sgpr, scratch, lds = meta[name]
        print(
            f"{100.0 * ns / total:7.2f} {ns / 1e6:10.3f} {n:7d} {ns / n / 1e3:9.2f} "
            f"{vgpr:>5} {sgpr:>5} {scratch:>6} {lds:>6}  {name[:90]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
