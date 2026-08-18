#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=col-r5
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/r5-%j.out
# Round 5: validate the upstream-shaped fix -- device prints compiled out behind a
# module constant (not deleted) plus the static octree index -- at kernel level,
# end to end, and for contact equivalence.
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export TMPDIR=$A/.tmp-r5-$SLURM_JOB_ID
mkdir -p $TMPDIR
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
source $A/venv/bin/activate
cd $A/mujoco_warp
ASSETS=/matx/u/knatalia/.bench_assets
SDF=$ASSETS/aloha_sdf/scene_sdf.xml
SRC=mujoco_warp/_src/collision_sdf.py
CB=$A/bin/collision_bench.py
EQ=$A/bin/sdf_equiv.py
export XDG_CACHE_HOME=$A/.cache-r5-$SLURM_JOB_ID
mkdir -p $XDG_CACHE_HOME

ks () { echo "########## R5K label=$1"; python $CB $SDF --nworld=8192 --nconmax=32 --njmax=226 \
  --label=$1 2>&1 | grep -E "^#|^RESULT|Traceback|Error|error"; }
e2e () { echo "########## R5E label=$1"; mjwarp-testspeed $SDF --format=short --event_trace=false \
  --nworld=8192 --nconmax=32 --njmax=226 --nstep=400 2>&1 | grep -E "steps_per_second|ncon_mean|solver_niter_mean"; }

stock ()  { cp $SRC.orig $SRC; }
guard ()  { cp $SRC.orig $SRC; python $A/sdf_debugprint_patch.py $SRC; }
guardoct(){ cp $SRC.orig $SRC; python $A/sdf_oct_patch.py $SRC; python $A/sdf_debugprint_patch.py $SRC; }

echo "=================== kernel level"
stock;    ks stock_a
guard;    ks guard
guardoct; ks guardoct
stock;    ks stock_b

echo "=================== end to end"
stock;    e2e stock_a
guard;    e2e guard
guardoct; e2e guardoct
stock;    e2e stock_b
guardoct; e2e guardoct_b

echo "=================== contact equivalence (512 worlds, 20 steps)"
stock;    python $EQ $SDF --nworld=512 --nconmax=32 --njmax=226 --steps=20 --out=$TMPDIR/stock.npz 2>&1 | tail -3
stock;    python $EQ $SDF --nworld=512 --nconmax=32 --njmax=226 --steps=20 --out=$TMPDIR/stock2.npz 2>&1 | tail -3
guardoct; python $EQ $SDF --nworld=512 --nconmax=32 --njmax=226 --steps=20 --out=$TMPDIR/go.npz 2>&1 | tail -3
echo "--- control: stock vs stock (run-to-run floor)"
python $EQ --compare $TMPDIR/stock.npz $TMPDIR/stock2.npz
echo "--- guarded+oct vs stock"
python $EQ --compare $TMPDIR/stock.npz $TMPDIR/go.npz

echo "=================== registers"
python $A/bin/hsaco_regs.py $XDG_CACHE_HOME/warp --filter collision_sdf
stock
echo "R5_DONE"
