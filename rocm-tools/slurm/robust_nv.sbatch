#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=robust-nv
#SBATCH --output=/matx/u/knatalia/warp-rocm-logs/robust-nv-%j.out
set -uxo pipefail
export SCRATCH=/matx/u/knatalia
export AGENT=$SCRATCH/agent-robust
export XDG_CACHE_HOME=$AGENT/.cache-nv-$SLURM_JOB_ID
export TMPDIR=$AGENT/.tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
# NVIDIA control: stock upstream Warp 1.16 (no ROCm patches)
source $SCRATCH/nvidia-baseline/venv/bin/activate
python -c "import warp; print('warp', warp.__version__)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "=============== PHASE 1: null-stream implicit sync probe ==============="
python $AGENT/tools/null_stream_sync.py --spin 200000 --trials 25
echo "PHASE1_RC=$?"

echo "=============== PHASE 2: captured cross-stream fork/join ==============="
python $AGENT/tools/capture_fork_join.py --spin 200000 --trials 20
echo "PHASE2_RC=$?"

echo "=============== PHASE 3: targeted copy repro (300 iters) ==============="
timeout 2400 python $AGENT/tools/copy_repro.py --variants target --iters 300
echo "PHASE3_RC=$?"

echo "=============== PHASE 4: copy variant sweep ==============="
timeout 3600 python $AGENT/tools/copy_repro.py --variants all --iters 20
echo "PHASE4_RC=$?"

echo "=============== PHASE 5: primitives launch dims on CUDA ==============="
python $AGENT/tools/big_launch.py
echo "PHASE5_RC=$?"

echo "ROBUST_NV_DONE"
