#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=robust-build
#SBATCH --output=/matx/u/knatalia/warp-rocm-logs/robust-build-%j.out
set -uxo pipefail
export SCRATCH=/matx/u/knatalia
export AGENT=$SCRATCH/agent-robust
export XDG_CACHE_HOME=$AGENT/.cache-build-$SLURM_JOB_ID
export PIP_CACHE_DIR=$AGENT/.pip-cache
export TMPDIR=$AGENT/.tmp
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
export ROCM_PATH=/opt/rocm
for v in 14 13 12; do
  if [ -f /usr/include/c++/$v/cmath ]; then GCCVER=$v; break; fi
done
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/$GCCVER"
export HIPCC_LINK_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/$GCCVER"
export PATH=$ROCM_PATH/bin:$PATH

# Isolated clone so the shared $SCRATCH/warp-rocm tree other agents use is untouched.
if [ ! -d $AGENT/warp ]; then
  git clone -q $SCRATCH/warp-rocm $AGENT/warp
fi
cd $AGENT/warp
git checkout -q rocm-117 || git checkout -q -b rocm-117 origin/rocm-117
git checkout -q -- warp/native/warp.cu
git apply $AGENT/hsa_grid_clamp.patch
git diff --stat

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -U pip numpy
python build_lib.py --rocm-path=/opt/rocm --hip-arch=gfx950 --no-use-libmathdx 2>&1 | tail -20
pip install -q -e .
python -c "import warp; warp.init(); print('warp', warp.__version__)"
echo "BUILD_DONE"

echo "=============== PHASE 1: oversized grid-stride launch ==============="
python $AGENT/tools/big_launch.py
echo "PHASE1_RC=$?"

echo "=============== PHASE 2: un-gated test_large.py on HIP ==============="
mkdir -p $AGENT/ungated
sed -e 's/^cuda_devices_below_uint32_launch = \[d for d in get_selected_cuda_test_devices() if not d.is_hip\]/cuda_devices_below_uint32_launch = get_selected_cuda_test_devices()/' \
    $AGENT/warp/warp/tests/test_large.py > $AGENT/ungated/test_large_ungated.py
grep -n "is_hip" $AGENT/ungated/test_large_ungated.py || echo "NO_HIP_GATES_LEFT"
timeout 3000 python $AGENT/ungated/test_large_ungated.py 2>&1 | tail -40
echo "PHASE2_RC=$?"

echo "=============== PHASE 3: primitives benchmark ==============="
export PRIM_ASSETS=$AGENT/.bench_assets
mkdir -p "$PRIM_ASSETS"
pip install -q --no-deps -e $SCRATCH/mujoco_warp
pip install -q absl-py "etils[epath]" "mujoco>=3.11.0" 2>&1 | tail -2
cd $SCRATCH/mujoco_warp
timeout 2400 mjwarp-testspeed $PRIM_ASSETS/primitives/primitives.xml \
  --clear_warp_cache=false --format=short --event_trace=false --memory=true \
  --measure_solver=true --measure_alloc=true \
  --function=render --nworld=8192 --nconmax=100 --njmax=256 --nstep=100 2>&1 | tail -40
echo "PHASE3_RC=$?"

echo "=============== PHASE 4: regression -- launch/graph/tile suites ==============="
cd $AGENT/warp
timeout 4000 python -m warp.tests -s autodetect -k TestLarge -k TestGraph -k TestLaunch -k TestArray -k TestAsync 2>&1 | tail -30
echo "PHASE4_RC=$?"

echo "ROBUST_BUILD_DONE"
