#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=col-nvab
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/nvab-%j.out
# Do the two mujoco_warp SDF fixes help, hurt, or do nothing on CUDA? Decides
# whether they are upstream candidates or ROCm-only workarounds.
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export XDG_CACHE_HOME=$A/.cache-nvab-$SLURM_JOB_ID
export TMPDIR=/tmp
mkdir -p $XDG_CACHE_HOME
nvidia-smi --query-gpu=name --format=csv,noheader
source $A/nv/venv/bin/activate
cd $A/nv/mujoco_warp
ASSETS=/matx/u/knatalia/.bench_assets
SDF=$ASSETS/aloha_sdf/scene_sdf.xml
SRC=mujoco_warp/_src/collision_sdf.py
CB=$A/bin/collision_bench.py
cp $SRC $SRC.orig 2>/dev/null || true

ks () { echo "########## NVK label=$1"; python $CB $SDF --nworld=8192 --nconmax=32 --njmax=226 \
  --label=$1 2>&1 | grep -E "^#|^RESULT|Traceback|Error|error"; }
e2e () { echo "########## NVE label=$1"; mjwarp-testspeed $SDF --format=short --event_trace=false \
  --nworld=8192 --nconmax=32 --njmax=226 --nstep=400 2>&1 | grep -E "steps_per_second|ncon_mean"; }

stock ()  { cp $SRC.orig $SRC; }
guardoct(){ cp $SRC.orig $SRC; python $A/sdf_oct_patch.py $SRC; python $A/sdf_debugprint_patch.py $SRC; }

stock;    ks nv_stock_a;  e2e nv_stock_a
guardoct; ks nv_guardoct; e2e nv_guardoct
stock;    ks nv_stock_b;  e2e nv_stock_b
stock
echo "NVAB_DONE"
