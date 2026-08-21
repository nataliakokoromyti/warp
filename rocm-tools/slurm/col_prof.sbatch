#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=col-prof
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/prof-%j.out
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export XDG_CACHE_HOME=$A/.cache-$SLURM_JOB_ID
export TMPDIR=$A/.tmp-$SLURM_JOB_ID
mkdir -p $XDG_CACHE_HOME $TMPDIR
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
source $A/venv/bin/activate
cd $A/mujoco_warp
ASSETS=/matx/u/knatalia/.bench_assets
OUT=$A/logs/prof-$SLURM_JOB_ID
mkdir -p $OUT

echo "=============== ENV"
python -c "import warp as wp; wp.init(); d=wp.get_device(); print(d, getattr(d,'arch',None))"

echo "=============== BASELINE aloha_sdf"
mjwarp-testspeed $ASSETS/aloha_sdf/scene_sdf.xml --format=short --event_trace=false \
  --nworld=8192 --nconmax=32 --njmax=226 --nstep=1000 2>&1 | tail -30

echo "=============== BASELINE aloha_clutter"
mjwarp-testspeed $ASSETS/aloha_clutter/scene_clutter.xml --format=short --event_trace=false \
  --nworld=2048 --nconmax=256 --nccdmax=16 --njmax=384 --nvmax=56 \
  --override opt.enableflags=SLEEP --init_asleep=true \
  --replay=$ASSETS/aloha_clutter/pick_clutter.npz --nstep=1000 2>&1 | tail -30

echo "=============== EVENT TRACE aloha_sdf"
mjwarp-testspeed $ASSETS/aloha_sdf/scene_sdf.xml --format=short --event_trace=true \
  --nworld=8192 --nconmax=32 --njmax=226 --nstep=200 2>&1 | tail -100

echo "=============== ROCPROF kernel trace aloha_sdf"
cd $OUT
rocprofv3 --kernel-trace --output-format csv -d $OUT/sdf -o sdftrace -- \
  mjwarp-testspeed $ASSETS/aloha_sdf/scene_sdf.xml --format=short --event_trace=false \
  --nworld=8192 --nconmax=32 --njmax=226 --nstep=60 2>&1 | tail -20
find $OUT/sdf -name "*.csv" | head

echo "=============== ROCPROF kernel trace aloha_clutter"
rocprofv3 --kernel-trace --output-format csv -d $OUT/clutter -o cluttertrace -- \
  mjwarp-testspeed $ASSETS/aloha_clutter/scene_clutter.xml --format=short --event_trace=false \
  --nworld=2048 --nconmax=256 --nccdmax=16 --njmax=384 --nvmax=56 \
  --override opt.enableflags=SLEEP --init_asleep=true \
  --replay=$ASSETS/aloha_clutter/pick_clutter.npz --nstep=60 2>&1 | tail -20
find $OUT/clutter -name "*.csv" | head

echo "=============== KERNEL CACHE"
ls $XDG_CACHE_HOME/warp 2>/dev/null | head
du -sh $XDG_CACHE_HOME
echo "PROF_DONE"
