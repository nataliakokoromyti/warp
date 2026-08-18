#!/bin/bash
# Decompose the in-capture allocation GPU fault.
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --job-name=robust-amd5
#SBATCH --output=/matx/u/knatalia/warp-rocm-logs/robust-amd5-%j.out
set -uo pipefail
export SCRATCH=/matx/u/knatalia
export AGENT=$SCRATCH/agent-robust
export XDG_CACHE_HOME=$AGENT/.cache5-$SLURM_JOB_ID
export TMPDIR=$AGENT/.tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd $SCRATCH/warp-rocm && source .venv/bin/activate
echo "=============== graph alloc fault decomposition ==============="
timeout 4000 python $AGENT/tools/graph_alloc_fault.py
echo "AMD5_SWEEP_END"
echo "ROBUST_AMD5_DONE"
