#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=col-ksweep
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/ksweep-%j.out
# Kernel-level sweep of _sdf_narrowphase launch config, measured with
# collision_bench (isolates the kernel, far less noise than end-to-end steps/s).
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export TMPDIR=$A/.tmp-ks-$SLURM_JOB_ID
mkdir -p $TMPDIR
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
source $A/venv/bin/activate
cd $A/mujoco_warp
ASSETS=/matx/u/knatalia/.bench_assets
CB=$A/bin/collision_bench.py
SRC=mujoco_warp/_src/collision_sdf.py
export XDG_CACHE_HOME=$A/.cache-ks-$SLURM_JOB_ID
mkdir -p $XDG_CACHE_HOME

ks () {  # $1 label, $2 block_dim, $3 min_blocks
  echo "########## KS label=$1 block_dim=$2 min_blocks=$3"
  MJW_SDF_BLOCK_DIM=$2 MJW_SDF_MIN_BLOCKS=$3 \
  python $CB $ASSETS/aloha_sdf/scene_sdf.xml --nworld=8192 --nconmax=32 --njmax=226 \
    --label=$1 2>&1 | grep -E "^#|^RESULT|Traceback|Error|error"
}

echo "===== VARIANT A: printf present"
cp $SRC.orig $SRC && python $A/apply_sdf_tune.py $SRC
ks ctrl_a0   0   1
ks lb64      64  1
ks lb128     128 1
ks lb256     256 1
ks lb512     512 1
ks lb1024    1024 1
ks lb256m2   256 2
ks ctrl_a1   0   1

echo "===== VARIANT B: device printf stripped"
cp $SRC.orig $SRC && python $A/apply_sdf_noprint.py $SRC && python $A/apply_sdf_tune.py $SRC
grep -c "device printf removed" $SRC
ks np_ctrl   0   1
ks np_lb256  256 1
ks np_lb64   64  1
ks np_ctrl2  0   1

cp $SRC.orig $SRC
echo "KSWEEP_DONE"
