#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=cbr-nv
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/cbrnv-%j.out
# Collision-phase comparison at the state the benchmark suite actually measures
# (mid-replay), not the settled keyframe pose.
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export XDG_CACHE_HOME=$A/.cache-cbrnv-$SLURM_JOB_ID
export TMPDIR=/tmp
mkdir -p $XDG_CACHE_HOME
nvidia-smi --query-gpu=name --format=csv,noheader
source $A/nv/venv/bin/activate
cd $A/nv/mujoco_warp
ASSETS=/matx/u/knatalia/.bench_assets
CB=$A/bin/collision_bench.py
G="^#|^RESULT|Traceback|Error|error"

python $CB $ASSETS/unitree_g1_hfield/scene_hfield.xml --nworld=8192 \
  --replay=$ASSETS/unitree_g1_hfield/shuffle_dance.npz --replay_step=300 --label=nvr_hfield 2>&1 | grep -E "$G"
python $CB $ASSETS/aloha_clutter/scene_clutter.xml --nworld=2048 --nconmax=256 --nccdmax=16 --njmax=384 --nvmax=56 \
  --override opt.enableflags=SLEEP --init_asleep \
  --replay=$ASSETS/aloha_clutter/pick_clutter.npz --replay_step=300 --label=nvr_clutter 2>&1 | grep -E "$G"
python $CB $ASSETS/unitree_g1_flat/scene_flat.xml --nworld=8192 \
  --replay=$ASSETS/unitree_g1_hfield/shuffle_dance.npz --replay_step=300 --label=nvr_g1flat 2>&1 | grep -E "$G"
echo "CBRNV_DONE"
