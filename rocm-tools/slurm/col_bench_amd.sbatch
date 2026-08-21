#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=col-bench
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/cb-%j.out
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export XDG_CACHE_HOME=$A/.cache-cb-$SLURM_JOB_ID
export TMPDIR=$A/.tmp-cb-$SLURM_JOB_ID
mkdir -p $XDG_CACHE_HOME $TMPDIR
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
source $A/venv/bin/activate
cd $A/mujoco_warp
ASSETS=/matx/u/knatalia/.bench_assets
CB=$A/bin/collision_bench.py
G="^#|^RESULT|Error|Traceback|error"

# make sure we measure stock sources here
cp mujoco_warp/_src/collision_sdf.py.orig mujoco_warp/_src/collision_sdf.py

python $CB $ASSETS/aloha_sdf/scene_sdf.xml --nworld=8192 --nconmax=32 --njmax=226 --label=amd_sdf 2>&1 | grep -E "$G"
python $CB $ASSETS/aloha_clutter/scene_clutter.xml --nworld=2048 --nconmax=256 --nccdmax=16 --njmax=384 --nvmax=56 --override opt.enableflags=SLEEP --init_asleep --label=amd_clutter 2>&1 | grep -E "$G"
python $CB $ASSETS/unitree_g1_hfield/scene_hfield.xml --nworld=8192 --label=amd_hfield 2>&1 | grep -E "$G"
python $CB $ASSETS/unitree_g1_flat/scene_flat.xml --nworld=8192 --label=amd_g1flat 2>&1 | grep -E "$G"
echo "CB_DONE"
