#!/bin/bash
# Build Warp into an isolated clone and validate the launch-limit fix against it.
#
# Isolated because the shared $SCRATCH/warp-rocm tree is used by other work: never
# rebuild native libraries there while someone else is running against them. Set
# PATCH=/path/to/x.patch to apply a working patch on top before building.
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=robust-build
#SBATCH --output=/matx/u/knatalia/warp-rocm-logs/robust-build-%j.out
set -uo pipefail
export SCRATCH=${SCRATCH:-/matx/u/knatalia}
export SRC=${SRC:-$SCRATCH/warp-rocm}
export BUILD_DIR=${BUILD_DIR:-$SCRATCH/agent-robust/warp}
export XDG_CACHE_HOME=$SCRATCH/.cache-build-${SLURM_JOB_ID:-manual}
export PIP_CACHE_DIR=$SCRATCH/.pip-cache
export TMPDIR=$SCRATCH/.tmp
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
# Node quirk: gcc-14 is installed without libstdc++-14-dev, so hipcc picks a GCC
# directory with no C++ headers. Point it at one that has them.
for v in 14 13 12; do
  if [ -f /usr/include/c++/$v/cmath ]; then GCCVER=$v; break; fi
done
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/$GCCVER"
export HIPCC_LINK_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/$GCCVER"
export PATH=$ROCM_PATH/bin:$PATH

[ -d "$BUILD_DIR" ] || git clone -q "$SRC" "$BUILD_DIR"
cd "$BUILD_DIR" || exit 1
git fetch -q origin
git checkout -q -B rocm-117 origin/rocm-117
[ -n "${PATCH:-}" ] && git apply "$PATCH"
git log --oneline -1

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -U pip numpy
# ~20 min: warp.cu is one large translation unit for hipcc.
python build_lib.py --rocm-path="$ROCM_PATH" --hip-arch=gfx950 --no-use-libmathdx 2>&1 | tail -20
pip install -q -e .
python -c "import warp; warp.init(); print('warp', warp.__version__)"
echo "BUILD_DONE"

echo "############ oversized grid-stride launches ############"
timeout 900 python "$BUILD_DIR/rocm-tools/big_launch.py"

echo "############ test_large.py un-gated on HIP ############"
# Both HIP-excluded tests launch past UINT32_MAX threads through grid-stride
# kernels and assert exact per-work-item counts, so they verify the clamp covers
# every element rather than merely not erroring.
UNGATED=$TMPDIR/ungated-build-${SLURM_JOB_ID:-manual}
mkdir -p "$UNGATED"
sed -e 's/^cuda_devices_below_uint32_launch = \[d for d in get_selected_cuda_test_devices() if not d.is_hip\]/cuda_devices_below_uint32_launch = get_selected_cuda_test_devices()/' \
    "$BUILD_DIR/warp/tests/test_large.py" > "$UNGATED/test_large_ungated.py"
grep -c is_hip "$UNGATED/test_large_ungated.py" || echo "NO_HIP_GATES_LEFT"
timeout 3000 python "$UNGATED/test_large_ungated.py" 2>&1 | tail -40

echo "############ primitives benchmark ############"
pip install -q absl-py "etils[epath]" "mujoco>=3.11.0" 2>&1 | tail -2
pip install -q --no-deps -e "$SCRATCH/mujoco_warp"
timeout 2400 mjwarp-testspeed "${ASSETS:-$SCRATCH/.bench_assets}/primitives/primitives.xml" \
  --clear_warp_cache=false --format=short --event_trace=false --memory=true \
  --measure_solver=true --measure_alloc=true \
  --function=render --nworld=8192 --nconmax=100 --njmax=256 --nstep=100 2>&1 | tail -40

echo "############ regression: launch/graph/array/copy suites ############"
cd "$BUILD_DIR"
timeout 4000 python -m warp.tests -s autodetect -k TestLarge -k TestGraph -k TestLaunch -k TestArray -k TestAsync 2>&1 | tail -30

echo "ROBUST_BUILD_DONE"
