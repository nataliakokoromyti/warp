#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=col-bench-nv
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/cbnv-%j.out
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export XDG_CACHE_HOME=$A/.cache-cbnv-$SLURM_JOB_ID
export TMPDIR=/tmp
mkdir -p $XDG_CACHE_HOME
nvidia-smi --query-gpu=name --format=csv,noheader
source $A/nv/venv/bin/activate
cd $A/nv/mujoco_warp
ASSETS=/matx/u/knatalia/.bench_assets
CB=$A/bin/collision_bench.py

python $CB $ASSETS/aloha_sdf/scene_sdf.xml --nworld=8192 --nconmax=32 --njmax=226 --label=nv_sdf 2>&1 | grep -E "^#|^RESULT|Error|Traceback|error"
python $CB $ASSETS/aloha_clutter/scene_clutter.xml --nworld=2048 --nconmax=256 --nccdmax=16 --njmax=384 --nvmax=56 --override opt.enableflags=SLEEP --init_asleep --label=nv_clutter 2>&1 | grep -E "^#|^RESULT|Error|Traceback|error"
python $CB $ASSETS/unitree_g1_hfield/scene_hfield.xml --nworld=8192 --label=nv_hfield 2>&1 | grep -E "^#|^RESULT|Error|Traceback|error"
python $CB $ASSETS/unitree_g1_flat/scene_flat.xml --nworld=8192 --label=nv_g1flat 2>&1 | grep -E "^#|^RESULT|Error|Traceback|error"
echo "CBNV_DONE"
