#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=col-r3
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/r3-%j.out
# Round 3: octree descent -- static-index the child vector (no dynamic register
# indexing) -- alone and combined with the launch_bounds fix. Kernel-level.
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export TMPDIR=$A/.tmp-r3-$SLURM_JOB_ID
mkdir -p $TMPDIR
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
source $A/venv/bin/activate
cd $A/mujoco_warp
ASSETS=/matx/u/knatalia/.bench_assets
CB=$A/bin/collision_bench.py
SRC=mujoco_warp/_src/collision_sdf.py
export XDG_CACHE_HOME=$A/.cache-r3-$SLURM_JOB_ID
mkdir -p $XDG_CACHE_HOME

ks () {  # $1 label, $2 block_dim
  echo "########## R3 label=$1 block_dim=$2"
  MJW_SDF_BLOCK_DIM=$2 MJW_SDF_MIN_BLOCKS=1 \
  python $CB $ASSETS/aloha_sdf/scene_sdf.xml --nworld=8192 --nconmax=32 --njmax=226 \
    --label=$1 2>&1 | grep -E "^#|^RESULT|Traceback|Error|error"
}

echo "===== BASE (stock body)"
cp $SRC.orig $SRC && python $A/apply_sdf_tune.py $SRC
ks base_ctrl 0
ks base_lb256 256

echo "===== OCT (static octree index)"
cp $SRC.orig $SRC && python $A/apply_sdf_oct.py $SRC && python $A/apply_sdf_tune.py $SRC
ks oct_ctrl 0
ks oct_lb256 256

echo "===== OCT + NOPRINT"
cp $SRC.orig $SRC && python $A/apply_sdf_oct.py $SRC && python $A/apply_sdf_noprint.py $SRC && python $A/apply_sdf_tune.py $SRC
ks octnp_ctrl 0
ks octnp_lb256 256

echo "===== BASE again (drift control)"
cp $SRC.orig $SRC && python $A/apply_sdf_tune.py $SRC
ks base_ctrl2 0
ks base_lb256b 256

cp $SRC.orig $SRC
echo "R3_DONE"
