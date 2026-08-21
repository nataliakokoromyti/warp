#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --job-name=col-sweep
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/sweep-%j.out
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export XDG_CACHE_HOME=$A/.cache-sw-$SLURM_JOB_ID
export TMPDIR=$A/.tmp-sw-$SLURM_JOB_ID
mkdir -p $XDG_CACHE_HOME $TMPDIR
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
source $A/venv/bin/activate
cd $A/mujoco_warp
python -c "import mujoco_warp; print('MJW:', mujoco_warp.__file__)"
ASSETS=/matx/u/knatalia/.bench_assets
OUT=$A/logs/sweep-$SLURM_JOB_ID
mkdir -p $OUT

# apply the tunable-launch patch to our own copy (idempotent: restore first)
cp mujoco_warp/_src/collision_sdf.py.orig mujoco_warp/_src/collision_sdf.py
python $A/apply_sdf_tune.py mujoco_warp/_src/collision_sdf.py
grep -n "_SDF_BLOCK_DIM\|launch_bounds=_SDF\|block_dim=_SDF" mujoco_warp/_src/collision_sdf.py

sdf_run () {   # $1 label, $2 block_dim, $3 min_blocks
  echo "########## SDFCFG label=$1 block_dim=$2 min_blocks=$3"
  MJW_SDF_BLOCK_DIM=$2 MJW_SDF_MIN_BLOCKS=$3 \
  mjwarp-testspeed $ASSETS/aloha_sdf/scene_sdf.xml --format=short --event_trace=false \
    --nworld=8192 --nconmax=32 --njmax=226 --nstep=1000 2>&1 | grep -E "steps_per_second|jit_duration|solver_niter_mean|ncon_mean|Error|error"
}

sdf_ktrace () { # $1 label, $2 block_dim, $3 min_blocks
  echo "########## SDFKT label=$1 block_dim=$2 min_blocks=$3"
  rm -rf $OUT/kt_$1
  MJW_SDF_BLOCK_DIM=$2 MJW_SDF_MIN_BLOCKS=$3 \
  rocprofv3 --kernel-trace --output-format csv -d $OUT/kt_$1 -o t -- \
    mjwarp-testspeed $ASSETS/aloha_sdf/scene_sdf.xml --format=short --event_trace=false \
    --nworld=8192 --nconmax=32 --njmax=226 --nstep=60 > /dev/null 2>&1
  C=$(find $OUT/kt_$1 -name "*kernel*.csv" | head -1)
  echo "CSV=$C"
  python $A/ktrace_top.py "$C" --top 12 2>&1 | head -20
}

# --- warm everything up once (full JIT) --------------------------------------
sdf_run warmup0 0 1

# --- controls and configs, interleaved to expose drift ------------------------
sdf_run ctrl_a 0 1
sdf_run lb256 256 1
sdf_run ctrl_b 0 1
sdf_run lb128 128 1
sdf_run lb64 64 1
sdf_run lb512 512 1
sdf_run lb256m2 256 2
sdf_run ctrl_c 0 1

# --- kernel-level confirmation ------------------------------------------------
sdf_ktrace ctrl 0 1
sdf_ktrace lb256 256 1
sdf_ktrace lb64 64 1

# --- register / scratch metadata ---------------------------------------------
echo "########## HSACO METADATA"
RD=/opt/rocm/llvm/bin/llvm-readelf
for f in $(find $XDG_CACHE_HOME -name "*.hsaco" -newermt "-8 hours" | xargs -r grep -l "sdf_narrowphase" 2>/dev/null | head -8); do
  echo "--- $f"
  $RD --notes "$f" 2>/dev/null | grep -E "vgpr_count|sgpr_count|private_segment_fixed_size|group_segment_fixed_size|max_flat_workgroup_size|\.name" | head -20
done
echo "SWEEP_DONE"
