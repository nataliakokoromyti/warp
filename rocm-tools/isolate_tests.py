"""Run every test in a unittest module in its own process.

A test that aborts the GPU (HSA memory access fault) kills the whole process and
hides every test after it, so an un-gating experiment needs one process per test.
Prints a PASS/FAIL/CRASH line per test and a summary.

    python isolate_tests.py path/to/test_module.py [--filter SUBSTRING]
"""

import argparse
import importlib.util
import subprocess
import sys
import unittest


def load_module(path):
    spec = importlib.util.spec_from_file_location("_isolate_target", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_isolate_target"] = module
    spec.loader.exec_module(module)
    return module


def collect_names(module):
    names = []
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
            for name in unittest.TestLoader().getTestCaseNames(obj):
                names.append(f"{attr}.{name}")
    return sorted(set(names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module")
    parser.add_argument("--filter", default="")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    module = load_module(args.module)
    names = [n for n in collect_names(module) if args.filter in n]
    print(f"discovered {len(names)} tests in {args.module}", flush=True)
    if args.list_only:
        for n in names:
            print(f"  {n}")
        return 0

    tally = {"PASS": 0, "FAIL": 0, "CRASH": 0, "SKIP": 0}
    for name in names:
        cmd = [sys.executable, args.module, name]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, check=False)
            out = proc.stdout + proc.stderr
            if proc.returncode == 0:
                verdict = "SKIP" if "skipped" in out and "Ran 1 test" in out and " ok" not in out else "PASS"
            elif proc.returncode < 0 or "Memory access fault" in out or "HSA_STATUS_ERROR" in out:
                verdict = "CRASH"
            else:
                verdict = "FAIL"
        except subprocess.TimeoutExpired:
            verdict = "CRASH"
            out = "timed out"
        tally[verdict] += 1
        print(f"{verdict:5s} {name}", flush=True)
        if verdict in ("FAIL", "CRASH"):
            tail = "\n".join(out.strip().splitlines()[-18:])
            print(f"------ {name} output ------\n{tail}\n------", flush=True)

    print(f"\nsummary: {tally}", flush=True)
    return 1 if tally["FAIL"] or tally["CRASH"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
