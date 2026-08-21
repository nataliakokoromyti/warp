#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=col-prof-nv
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/profnv-%j.out
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export XDG_CACHE_HOME=$A/.cache-nv-$SLURM_JOB_ID
export TMPDIR=/tmp
mkdir -p $XDG_CACHE_HOME
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
source $A/nv/venv/bin/activate
python -c "import mujoco_warp, warp as wp; print('mjw:', mujoco_warp.__file__); print('warp:', wp.__version__)"
cd $A/nv/mujoco_warp
ASSETS=/matx/u/knatalia/.bench_assets
OUT=$A/logs/profnv-$SLURM_JOB_ID
mkdir -p $OUT

echo "=============== BASELINE aloha_sdf"
mjwarp-testspeed $ASSETS/aloha_sdf/scene_sdf.xml --format=short --event_trace=false \
  --nworld=8192 --nconmax=32 --njmax=226 --nstep=1000 2>&1 | tail -30

echo "=============== BASELINE aloha_clutter"
mjwarp-testspeed $ASSETS/aloha_clutter/scene_clutter.xml --format=short --event_trace=false \
  --nworld=2048 --nconmax=256 --nccdmax=16 --njmax=384 --nvmax=56 \
  --override opt.enableflags=SLEEP --init_asleep=true \
  --replay=$ASSETS/aloha_clutter/pick_clutter.npz --nstep=1000 2>&1 | tail -30

which nsys || echo "NO NSYS"
cd $OUT
if which nsys; then
echo "=============== NSYS aloha_sdf"
nsys profile -t cuda --cuda-graph-trace=node -f true -o $OUT/sdf \
  mjwarp-testspeed $ASSETS/aloha_sdf/scene_sdf.xml --format=short --event_trace=false \
  --nworld=8192 --nconmax=32 --njmax=226 --nstep=60 2>&1 | tail -10
nsys stats --report cuda_gpu_kern_sum --format csv $OUT/sdf.nsys-rep 2>&1 | head -40

echo "=============== NSYS aloha_clutter"
nsys profile -t cuda --cuda-graph-trace=node -f true -o $OUT/clutter \
  mjwarp-testspeed $ASSETS/aloha_clutter/scene_clutter.xml --format=short --event_trace=false \
  --nworld=2048 --nconmax=256 --nccdmax=16 --njmax=384 --nvmax=56 \
  --override opt.enableflags=SLEEP --init_asleep=true \
  --replay=$ASSETS/aloha_clutter/pick_clutter.npz --nstep=60 2>&1 | tail -10
nsys stats --report cuda_gpu_kern_sum --format csv $OUT/clutter.nsys-rep 2>&1 | head -40
fi

echo "=============== PTX/SASS register counts for sdf kernel"
find $XDG_CACHE_HOME -name "*.ptx" | head -3
echo "NVPROF_DONE"
