#!/bin/bash
# One-command ROCm validation gate.
#
# Replaces the ad-hoc sbatch chain with a single job that builds Warp, runs the
# Warp and mujoco_warp suites, repeats the historically flaky test classes to
# catch intermittents, and smoke-tests the benchmark suite -- then prints a
# machine-greppable PASS/FAIL summary at the end.
#
#   sbatch rocm-tools/slurm/validate.sbatch            # full gate (~90 min)
#   VALIDATE_QUICK=1 sbatch rocm-tools/slurm/validate.sbatch   # skip benchmarks
#
# Override WARP_DIR / MJW_DIR to validate a tree other than the defaults.
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=rocm-validate
#SBATCH --output=/matx/u/knatalia/warp-rocm-logs/validate-%j.out
set -uo pipefail
export SCRATCH=${SCRATCH:-/matx/u/knatalia}
export WARP_DIR=${WARP_DIR:-$SCRATCH/warp-rocm}
export MJW_DIR=${MJW_DIR:-$SCRATCH/mujoco_warp}
export ASSETS=${ASSETS:-$SCRATCH/.bench_assets}
# Per-job kernel cache: concurrent jobs sharing one Warp cache corrupt each other.
export XDG_CACHE_HOME=$SCRATCH/.cache-validate-${SLURM_JOB_ID:-manual}
export PIP_CACHE_DIR=$SCRATCH/.pip-cache
export TMPDIR=$SCRATCH/.tmp
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export PATH=$ROCM_PATH/bin:$PATH
for v in 14 13 12; do
  if [ -f /usr/include/c++/$v/cmath ]; then GCCVER=$v; break; fi
done
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/$GCCVER"
export HIPCC_LINK_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/$GCCVER"

RESULTS=()
record() { RESULTS+=("$1 $2"); echo "### GATE $1 $2"; }

cd "$WARP_DIR" || exit 1
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate

echo "############ 1/5 build ############"
if python build_lib.py --rocm-path="$ROCM_PATH" --hip-arch=gfx950 --no-use-libmathdx 2>&1 | tail -15; then
  pip install -q -e . && record build PASS
else
  record build FAIL
fi

echo "############ 2/5 warp suite ############"
python -m warp.tests -s autodetect 2>&1 | tail -25 | tee "$TMPDIR/warp-suite.txt"
if grep -qE "^OK( \(|$)" "$TMPDIR/warp-suite.txt"; then record warp_suite PASS; else record warp_suite FAIL; fi

echo "############ 3/5 flake watch (historically intermittent classes) ############"
# Both known intermittents were torn device-to-host reads, which only surface
# under repetition; a single green run does not clear them.
FLAKE_FAILS=0
for i in 1 2 3 4 5; do
  python -m warp.tests -s autodetect -k TestAsync -k TestFemField 2>&1 | tail -4 | tee "$TMPDIR/flake-$i.txt"
  grep -qE "^OK( \(|$)" "$TMPDIR/flake-$i.txt" || FLAKE_FAILS=$((FLAKE_FAILS + 1))
done
if [ "$FLAKE_FAILS" -eq 0 ]; then record flake_watch PASS; else record flake_watch "FAIL($FLAKE_FAILS/5)"; fi

echo "############ 4/5 mujoco_warp suite ############"
pip install -q absl-py "etils[epath]" "mujoco>=3.11.0" pytest pytest-timeout 2>&1 | tail -2
pip install -q --no-deps -e "$MJW_DIR"
cd "$MJW_DIR" || exit 1
python -m pytest mujoco_warp -q --timeout=600 2>&1 | tail -12 | tee "$TMPDIR/mjw.txt"
if grep -qE "[0-9]+ passed" "$TMPDIR/mjw.txt" && ! grep -qE "[0-9]+ failed" "$TMPDIR/mjw.txt"; then
  record mjw_suite PASS
else
  record mjw_suite FAIL
fi

if [ "${VALIDATE_QUICK:-0}" = "1" ]; then
  record benchmarks SKIPPED
else
  echo "############ 5/5 benchmark suite ############"
  python benchmarks/run.py --assets_root="$ASSETS" --clear_warp_cache=false 2>&1 | tail -40 | tee "$TMPDIR/bench.txt"
  if grep -q "BENCHMARK_FAILED" "$TMPDIR/bench.txt"; then record benchmarks FAIL; else record benchmarks PASS; fi
fi

echo
echo "================ VALIDATION SUMMARY ================"
printf '%s\n' "${RESULTS[@]}"
printf '%s\n' "${RESULTS[@]}" | grep -q FAIL && echo "OVERALL FAIL" || echo "OVERALL PASS"
echo "VALIDATE_DONE"
