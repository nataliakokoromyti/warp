#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --job-name=warp-lb
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/warplb-%j.out
# A/B a Warp codegen change: emit __launch_bounds__(WP_TILE_BLOCK_DIM) on HIP for
# kernels that declare none. Alternates OFF/ON so drift is visible.
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export TMPDIR=$A/.tmp-wlb-$SLURM_JOB_ID
mkdir -p $TMPDIR
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
source $A/lb/venv/bin/activate
cd $A/lb/mujoco_warp
cp mujoco_warp/_src/collision_sdf.py.orig mujoco_warp/_src/collision_sdf.py   # stock mjw
ASSETS=/matx/u/knatalia/.bench_assets
CG=$A/lb/warp-rocm/warp/_src/codegen.py

set_variant () {  # on|off
  if [ "$1" = on ]; then cp $A/lb/warp-rocm/warp/_src/codegen.new.py $CG
  else cp $A/lb/warp-rocm/warp/_src/codegen.py.orig $CG; fi
  grep -c "WP_DEFAULT_LAUNCH_BOUNDS" $CG
}

# each variant gets its own kernel cache: the generated source differs, and we do
# not want either variant to inherit the other's binaries.
run () {  # $1 variant, $2 scene-label, rest: testspeed args
  V=$1; L=$2; shift 2
  set_variant $V
  export XDG_CACHE_HOME=$A/.cache-wlb-$SLURM_JOB_ID-$V
  mkdir -p $XDG_CACHE_HOME
  echo "########## WLB variant=$V scene=$L"
  mjwarp-testspeed "$@" 2>&1 | grep -E "steps_per_second|ncon_mean|solver_niter_mean|Error|error:"
}

SDF="$ASSETS/aloha_sdf/scene_sdf.xml --format=short --event_trace=false --nworld=8192 --nconmax=32 --njmax=226 --nstep=400"
CLU="$ASSETS/aloha_clutter/scene_clutter.xml --format=short --event_trace=false --nworld=2048 --nconmax=256 --nccdmax=16 --njmax=384 --nvmax=56 --override opt.enableflags=SLEEP --init_asleep=true --replay=$ASSETS/aloha_clutter/pick_clutter.npz --nstep=600"
HF="$ASSETS/unitree_g1_hfield/scene_hfield.xml --format=short --event_trace=false --nworld=8192 --replay=$ASSETS/unitree_g1_hfield/shuffle_dance.npz"
FLAT="$ASSETS/unitree_g1_flat/scene_flat.xml --format=short --event_trace=false --nworld=8192 --replay=$ASSETS/unitree_g1_hfield/shuffle_dance.npz"

for pass in 1 2; do
  for v in off on; do
    run $v sdf_pass$pass  $SDF
    run $v clutter_pass$pass $CLU
    run $v hfield_pass$pass $HF
    run $v g1flat_pass$pass $FLAT
  done
done

set_variant off
echo "WARPLB_DONE"
