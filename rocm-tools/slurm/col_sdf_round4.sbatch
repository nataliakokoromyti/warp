#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=col-r4
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/r4-%j.out
# Round 4: is removing the SDF error printfs legitimate?
#   (a) do the device prints ever fire?  (unfiltered stdout)
#   (b) do contacts change?              (npz diff, stock vs no-print)
#   (c) what is it worth end to end?     (testspeed, controls interleaved)
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export TMPDIR=$A/.tmp-r4-$SLURM_JOB_ID
mkdir -p $TMPDIR
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
source $A/venv/bin/activate
cd $A/mujoco_warp
ASSETS=/matx/u/knatalia/.bench_assets
SDF=$ASSETS/aloha_sdf/scene_sdf.xml
SRC=mujoco_warp/_src/collision_sdf.py
EQ=$A/bin/sdf_equiv.py
export XDG_CACHE_HOME=$A/.cache-r4-$SLURM_JOB_ID
mkdir -p $XDG_CACHE_HOME

echo "=================== (a) do the device error prints fire? (stock, UNFILTERED)"
cp $SRC.orig $SRC
python $EQ $SDF --nworld=512 --nconmax=32 --njmax=226 --steps=20 --out=$TMPDIR/stock.npz 2>&1 | tee $TMPDIR/stock.log | tail -20
echo "--- count of device ERROR lines:"
grep -c "ERROR" $TMPDIR/stock.log || echo 0
grep -m5 "ERROR" $TMPDIR/stock.log || true

echo "=================== (b) contacts with the prints removed"
cp $SRC.orig $SRC && python $A/apply_sdf_noprint.py $SRC
python $EQ $SDF --nworld=512 --nconmax=32 --njmax=226 --steps=20 --out=$TMPDIR/nopr.npz 2>&1 | tail -5
echo "--- stock vs no-print"
python $EQ --compare $TMPDIR/stock.npz $TMPDIR/nopr.npz

echo "--- control: stock vs a second stock run (run-to-run floor)"
cp $SRC.orig $SRC
python $EQ $SDF --nworld=512 --nconmax=32 --njmax=226 --steps=20 --out=$TMPDIR/stock2.npz 2>&1 | tail -3
python $EQ --compare $TMPDIR/stock.npz $TMPDIR/stock2.npz

echo "=================== (c) end-to-end value"
run () { echo "########## R4 label=$1"; mjwarp-testspeed $SDF --format=short --event_trace=false \
  --nworld=8192 --nconmax=32 --njmax=226 --nstep=400 2>&1 | grep -E "steps_per_second|ncon_mean|solver_niter_mean"; }

cp $SRC.orig $SRC
run stock_a
cp $SRC.orig $SRC && python $A/apply_sdf_noprint.py $SRC
run noprint_a
cp $SRC.orig $SRC
run stock_b
cp $SRC.orig $SRC && python $A/apply_sdf_noprint.py $SRC
run noprint_b
cp $SRC.orig $SRC
run stock_c

echo "=================== register/spill for each variant"
python $A/bin/hsaco_regs.py $XDG_CACHE_HOME/warp --filter collision_sdf
echo "R4_DONE"
