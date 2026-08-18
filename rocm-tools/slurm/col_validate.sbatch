#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=col-val
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/val-%j.out
# Correctness gate for the two candidate changes:
#   (1) Warp codegen: default __launch_bounds__ on HIP  -> full Warp test suite
#   (2) mujoco_warp: static-indexed octree descent      -> mujoco_warp suite
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export TMPDIR=$A/.tmp-val-$SLURM_JOB_ID
mkdir -p $TMPDIR
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"

echo "=================== (1) Warp suite with default launch_bounds ON"
source $A/lb/venv/bin/activate
cp $A/lb/warp-rocm/warp/_src/codegen.new.py $A/lb/warp-rocm/warp/_src/codegen.py
cp $A/lb/mujoco_warp/mujoco_warp/_src/collision_sdf.py.orig $A/lb/mujoco_warp/mujoco_warp/_src/collision_sdf.py
export XDG_CACHE_HOME=$A/.cache-val-warp-$SLURM_JOB_ID
mkdir -p $XDG_CACHE_HOME
cd $A/lb/warp-rocm
python -m warp.tests -s autodetect 2>&1 | tail -25
echo "WARP_SUITE_DONE"

echo "=================== (2) mujoco_warp collision tests, octree patch"
deactivate || true
source $A/venv/bin/activate
cd $A/mujoco_warp
SRC=mujoco_warp/_src/collision_sdf.py
export XDG_CACHE_HOME=$A/.cache-val-mjw-$SLURM_JOB_ID
mkdir -p $XDG_CACHE_HOME

echo "--- baseline (stock collision_sdf.py)"
cp $SRC.orig $SRC
python -m pytest mujoco_warp/_src/collision_driver_test.py mujoco_warp/_src/collision_gjk_test.py -q 2>&1 | tail -8

echo "--- with static-indexed octree descent"
cp $SRC.orig $SRC && python $A/apply_sdf_oct.py $SRC
python -m pytest mujoco_warp/_src/collision_driver_test.py mujoco_warp/_src/collision_gjk_test.py -q 2>&1 | tail -8

echo "--- full mujoco_warp suite with the octree patch"
python -m pytest mujoco_warp -q -x --timeout=1200 2>&1 | tail -15
cp $SRC.orig $SRC
echo "VAL_DONE"
