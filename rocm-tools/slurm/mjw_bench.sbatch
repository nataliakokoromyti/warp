#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=mjw-bench
#SBATCH --output=/matx/u/knatalia/warp-rocm-logs/bench-%j.out
set -uxo pipefail
export SCRATCH=/matx/u/knatalia
export XDG_CACHE_HOME=$SCRATCH/.cache
export TMPDIR=$SCRATCH/.tmp
cd $SCRATCH/warp-rocm && source .venv/bin/activate
cd $SCRATCH/mujoco_warp
python3 benchmarks/run.py --assets_root=$SCRATCH/.bench_assets 2>&1 | tee $SCRATCH/warp-rocm-logs/bench-results-$SLURM_JOB_ID.log | tail -60
echo "BENCH_DONE"
