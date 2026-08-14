#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=warp-build
#SBATCH --output=/matx/u/knatalia/warp-rocm-logs/build-%j.out
set -euxo pipefail
export SCRATCH=/matx/u/knatalia
export XDG_CACHE_HOME=$SCRATCH/.cache
export PIP_CACHE_DIR=$SCRATCH/.pip-cache
export TMPDIR=$SCRATCH/.tmp
mkdir -p $XDG_CACHE_HOME $PIP_CACHE_DIR $TMPDIR
export ROCM_PATH=/opt/rocm
# pick a GCC install dir that actually ships libstdc++ headers
for v in 14 13 12; do
  if [ -f /usr/include/c++/$v/cmath ]; then GCCVER=$v; break; fi
done
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/$GCCVER"
export HIPCC_LINK_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/$GCCVER"
echo "using GCC $GCCVER for hipcc"

export PATH=$ROCM_PATH/bin:$PATH
cd $SCRATCH/warp-rocm && git pull -q origin rocm-backend
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -U pip numpy
python build_lib.py --rocm-path=/opt/rocm --hip-arch=gfx950 --no-use-libmathdx 2>&1 | tail -30
pip install -q -e .
echo "=== SMOKE TEST ==="
python /matx/u/knatalia/warp_smoke.py
echo "BUILD_AND_SMOKE_SUCCESS"
