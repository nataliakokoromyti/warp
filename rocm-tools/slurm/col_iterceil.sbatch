#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=col-ceil
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/ceil-%j.out
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export XDG_CACHE_HOME=$A/.cache-ce-$SLURM_JOB_ID
export TMPDIR=$A/.tmp-ce-$SLURM_JOB_ID
mkdir -p $XDG_CACHE_HOME $TMPDIR
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
source $A/venv/bin/activate
cd $A/mujoco_warp
ASSETS=/matx/u/knatalia/.bench_assets

echo "=============== configured opt.iterations per scene"
python - <<'PY'
import mujoco, os
A = "/matx/u/knatalia/.bench_assets"
for name, xml in [
    ("aloha_sdf", "aloha_sdf/scene_sdf.xml"),
    ("aloha_clutter", "aloha_clutter/scene_clutter.xml"),
    ("unitree_g1_hfield", "unitree_g1_hfield/scene_hfield.xml"),
]:
    m = mujoco.MjModel.from_xml_path(os.path.join(A, xml))
    print(f"{name}: iterations={m.opt.iterations} ls_iterations={m.opt.ls_iterations} "
          f"solver={m.opt.solver} cone={m.opt.cone} ngeom={m.ngeom}")
PY

sdf () {  # $1 label, extra args...
  L=$1; shift
  echo "########## CEIL scene=aloha_sdf label=$L args=$*"
  mjwarp-testspeed $ASSETS/aloha_sdf/scene_sdf.xml --format=short --event_trace=false \
    --nworld=8192 --nconmax=32 --njmax=226 --nstep=600 "$@" 2>&1 | grep -E "steps_per_second|solver_niter_mean|Error|error:"
}
clu () {
  L=$1; shift
  echo "########## CEIL scene=aloha_clutter label=$L args=$*"
  mjwarp-testspeed $ASSETS/aloha_clutter/scene_clutter.xml --format=short --event_trace=false \
    --nworld=2048 --nconmax=256 --nccdmax=16 --njmax=384 --nvmax=56 \
    --override opt.enableflags=SLEEP --init_asleep=true \
    --replay=$ASSETS/aloha_clutter/pick_clutter.npz --nstep=600 "$@" 2>&1 | grep -E "steps_per_second|solver_niter_mean|Error|error:"
}

sdf default
sdf iter1 -o opt.iterations=1
sdf iter4 -o opt.iterations=4
sdf default2
clu default
clu iter1 -o opt.iterations=1
clu default2

echo "=============== event trace aloha_sdf (aggregated by phase)"
mjwarp-testspeed $ASSETS/aloha_sdf/scene_sdf.xml --format=short --event_trace=true \
  --nworld=8192 --nconmax=32 --njmax=226 --nstep=200 > $A/logs/etrace_sdf_$SLURM_JOB_ID.txt 2>&1
python $A/etrace_agg.py $A/logs/etrace_sdf_$SLURM_JOB_ID.txt

echo "=============== event trace aloha_clutter (aggregated by phase)"
mjwarp-testspeed $ASSETS/aloha_clutter/scene_clutter.xml --format=short --event_trace=true \
  --nworld=2048 --nconmax=256 --nccdmax=16 --njmax=384 --nvmax=56 \
  --override opt.enableflags=SLEEP --init_asleep=true \
  --replay=$ASSETS/aloha_clutter/pick_clutter.npz --nstep=200 > $A/logs/etrace_clu_$SLURM_JOB_ID.txt 2>&1
python $A/etrace_agg.py $A/logs/etrace_clu_$SLURM_JOB_ID.txt
echo "CEIL_DONE"
