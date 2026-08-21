# Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Aggregate ``mjwarp-testspeed --event_trace=true`` output into phase totals.

The raw trace prints one line per event-scope instance
(``step.forward.solve._solver_iteration._linesearch.mul_m[317]: 0.0``), which is
thousands of lines. This rolls them up by scope prefix so the phase split is
readable.

Usage: python etrace_agg.py <testspeed_output.txt> [--depth N]
"""

import argparse
import re
import sys
from collections import defaultdict

LINE = re.compile(r"^(?P<scope>[A-Za-z_][\w.]*)\[(?P<idx>\d+)\]:\s+(?P<ms>[-\d.eE+]+)\s*$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--depth", type=int, default=3, help="scope depth to roll up to")
    args = ap.parse_args()

    totals = defaultdict(float)
    counts = defaultdict(int)
    leaf = defaultdict(float)
    grand = 0.0
    for line in open(args.path):
        m = LINE.match(line.strip())
        if not m:
            continue
        scope = m.group("scope")
        ms = float(m.group("ms"))
        grand += ms
        parts = scope.split(".")
        totals[".".join(parts[: args.depth])] += ms
        counts[".".join(parts[: args.depth])] += 1
        leaf[scope] += ms

    if not grand:
        print("no event-trace lines found (is --event_trace=true set?)")
        return 1

    print(f"ETRACE total={grand:.3f} ms over all recorded events")
    print(f"\n{'%':>7} {'ms':>10} {'n':>6}  scope (depth {args.depth})")
    for k, v in sorted(totals.items(), key=lambda kv: -kv[1])[:20]:
        print(f"{100 * v / grand:7.2f} {v:10.3f} {counts[k]:6d}  {k}")

    print(f"\n{'%':>7} {'ms':>10}  leaf scope")
    for k, v in sorted(leaf.items(), key=lambda kv: -kv[1])[:25]:
        print(f"{100 * v / grand:7.2f} {v:10.3f}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
