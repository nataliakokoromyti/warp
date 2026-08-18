#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=col-valm
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/valm-%j.out
# mujoco_warp suite, stock vs the two SDF fixes (guarded device prints + static
# octree index). Output is teed to a file so mujoco's own C-level stdout noise
# cannot bury the pytest summary.
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export TMPDIR=$A/.tmp-valm-$SLURM_JOB_ID
mkdir -p $TMPDIR
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
source $A/venv/bin/activate
cd $A/mujoco_warp
SRC=mujoco_warp/_src/collision_sdf.py
export XDG_CACHE_HOME=$A/.cache-valm-$SLURM_JOB_ID
mkdir -p $XDG_CACHE_HOME

suite () {   # $1 label
  L=$1
  F=$A/logs/mjwsuite-$SLURM_JOB_ID-$L.log
  echo "=================== mujoco_warp suite: $L"
  python -m pytest mujoco_warp -q -p no:randomly > $F 2>&1 || true
  echo "--- summary"
  grep -E "^[0-9]+ (passed|failed)|passed|failed|error" $F | tail -3
  echo "--- failures (if any)"
  grep -E "^FAILED |^ERROR " $F | head -20 || true
}

echo "### variant: stock"
cp $SRC.orig $SRC
suite stock

echo "### variant: guarded prints + static octree index"
cp $SRC.orig $SRC
python $A/sdf_oct_patch.py $SRC
python $A/sdf_debugprint_patch.py $SRC
suite patched

cp $SRC.orig $SRC
echo "VALM_DONE"
