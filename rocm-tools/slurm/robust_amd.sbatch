#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=robust-amd
#SBATCH --output=/matx/u/knatalia/warp-rocm-logs/robust-amd-%j.out
set -uxo pipefail
export SCRATCH=/matx/u/knatalia
export AGENT=$SCRATCH/agent-robust
export XDG_CACHE_HOME=$AGENT/.cache-$SLURM_JOB_ID
export TMPDIR=$AGENT/.tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd $SCRATCH/warp-rocm && source .venv/bin/activate
python -c "import warp; print('warp', warp.__version__)"

echo "=============== PHASE 1: captured cross-stream fork/join ==============="
python $AGENT/tools/capture_fork_join.py --spin 200000 --trials 20
echo "PHASE1_RC=$?"

echo "=============== PHASE 2: targeted copy repro (200 iters) ==============="
timeout 1800 python $AGENT/tools/copy_repro.py --variants target --iters 200
echo "PHASE2_RC=$?"

echo "=============== PHASE 3: copy variant sweep ==============="
timeout 3600 python $AGENT/tools/copy_repro.py --variants all --iters 20
echo "PHASE3_RC=$?"

echo "=============== PHASE 4: fem implicit fields x500 ==============="
timeout 1800 python $AGENT/tools/flake_hunt.py --test fem_implicit --iters 500
echo "PHASE4_RC=$?"

echo "=============== PHASE 5: fem implicit fields x300 under load ==============="
timeout 1800 python $AGENT/tools/flake_hunt.py --test fem_implicit --iters 300 --load 3
echo "PHASE5_RC=$?"

echo "=============== PHASE 6: primitives benchmark ==============="
export PRIM_ASSETS=$AGENT/.bench_assets
mkdir -p "$PRIM_ASSETS"
cd $SCRATCH/mujoco_warp
timeout 1800 python benchmarks/run.py -f primitives --assets_root=$PRIM_ASSETS --clear_warp_cache=false 2>&1 | tail -60
echo "PHASE6_RC=$?"
echo "--- direct mjwarp-testspeed invocation for full traceback ---"
timeout 1800 mjwarp-testspeed $PRIM_ASSETS/primitives/primitives.xml \
  --clear_warp_cache=false --format=short --event_trace=false --memory=true \
  --measure_solver=true --measure_alloc=true \
  --function=render --nworld=8192 --nconmax=100 --njmax=256 --nstep=100 2>&1 | tail -80
echo "PHASE6B_RC=$?"

echo "ROBUST_AMD_DONE"
