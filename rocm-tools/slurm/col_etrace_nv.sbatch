#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=col-et-nv
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/etnv-%j.out
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export XDG_CACHE_HOME=$A/.cache-etnv-$SLURM_JOB_ID
export TMPDIR=/tmp
mkdir -p $XDG_CACHE_HOME
nvidia-smi --query-gpu=name --format=csv,noheader
source $A/nv/venv/bin/activate
cd $A/nv/mujoco_warp
ASSETS=/matx/u/knatalia/.bench_assets

echo "=============== EVENT TRACE aloha_sdf (L40S)"
mjwarp-testspeed $ASSETS/aloha_sdf/scene_sdf.xml --format=short --event_trace=true \
  --nworld=8192 --nconmax=32 --njmax=226 --nstep=200 2>&1 | tail -120

echo "=============== EVENT TRACE aloha_clutter (L40S)"
mjwarp-testspeed $ASSETS/aloha_clutter/scene_clutter.xml --format=short --event_trace=true \
  --nworld=2048 --nconmax=256 --nccdmax=16 --njmax=384 --nvmax=56 \
  --override opt.enableflags=SLEEP --init_asleep=true \
  --replay=$ASSETS/aloha_clutter/pick_clutter.npz --nstep=200 2>&1 | tail -120

echo "=============== iterations ceiling aloha_sdf (L40S control: should be ~1.0)"
for it in 0 1; do
  echo "--- iterations override=$it"
  mjwarp-testspeed $ASSETS/aloha_sdf/scene_sdf.xml --format=short --event_trace=false \
    --nworld=8192 --nconmax=32 --njmax=226 --nstep=500 -o opt.iterations=$it 2>&1 | grep -E "steps_per_second|Error" || true
done
echo "ETNV_DONE"
