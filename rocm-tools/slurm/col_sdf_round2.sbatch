#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --job-name=col-r2
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/r2-%j.out
# Round 2: does device printf (HIP hostcall) cost us in the SDF kernel, and does
# it compound with the launch_bounds fix? BD is the winning block_dim from round 1.
set -uxo pipefail
BD=${BD:-256}
A=/matx/u/knatalia/agent-collision
export XDG_CACHE_HOME=$A/.cache-r2-$SLURM_JOB_ID
export TMPDIR=$A/.tmp-r2-$SLURM_JOB_ID
mkdir -p $XDG_CACHE_HOME $TMPDIR
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
source $A/venv/bin/activate
cd $A/mujoco_warp
ASSETS=/matx/u/knatalia/.bench_assets
SRC=mujoco_warp/_src/collision_sdf.py

run () {   # $1 label, $2 block_dim
  echo "########## R2 label=$1 block_dim=$2 BD=$BD"
  MJW_SDF_BLOCK_DIM=$2 MJW_SDF_MIN_BLOCKS=1 \
  mjwarp-testspeed $ASSETS/aloha_sdf/scene_sdf.xml --format=short --event_trace=false \
    --nworld=8192 --nconmax=32 --njmax=226 --nstep=400 2>&1 | grep -E "steps_per_second|ncon_mean|solver_niter_mean|Error|error:"
}

echo "===== VARIANT: stock + tunable launch (printf present)"
cp $SRC.orig $SRC && python $A/apply_sdf_tune.py $SRC
run warm 0
run ctrl_a 0
run lb$BD $BD
run ctrl_b 0

echo "===== VARIANT: device printf stripped"
cp $SRC.orig $SRC && python $A/apply_sdf_noprint.py $SRC && python $A/apply_sdf_tune.py $SRC
grep -c "device printf removed" $SRC
run np_ctrl 0
run np_lb$BD $BD
run np_ctrl2 0

echo "===== restore"
cp $SRC.orig $SRC
echo "R2_DONE"
