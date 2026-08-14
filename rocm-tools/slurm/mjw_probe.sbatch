#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=mjw-probe
#SBATCH --output=/matx/u/knatalia/warp-rocm-logs/mjw-probe-%j.out
set -uxo pipefail
export SCRATCH=/matx/u/knatalia
export XDG_CACHE_HOME=$SCRATCH/.cache
export PIP_CACHE_DIR=$SCRATCH/.pip-cache
export TMPDIR=$SCRATCH/.tmp
cd $SCRATCH/warp-rocm && source .venv/bin/activate
pip install -q absl-py "etils[epath]" "mujoco>=3.11.0" pytest pytest-timeout 2>&1 | tail -2
pip install -q --no-deps -e $SCRATCH/mujoco_warp
python -c "import warp; print(\"warp:\", warp.__version__)"
python -c "import mujoco_warp" && echo "IMPORT_OK" || echo "IMPORT_FAILED"
cd $SCRATCH/mujoco_warp
python -m pytest mujoco_warp -x -q --timeout=600 2>&1 | tail -40 || true
echo "=== full run without -x, collecting failure inventory ==="
python -m pytest mujoco_warp -q --timeout=600 2>&1 | tee $SCRATCH/warp-rocm-logs/mjw-pytest-$SLURM_JOB_ID.log | tail -30
echo "MJW_PROBE_DONE"
echo "=== DETAILED TRACEBACKS for remaining failures ==="
python -m pytest -v --tb=long --timeout=600 mujoco_warp/_src/sleep_test.py mujoco_warp/_src/island_test.py mujoco_warp/_src/unroll_test.py 2>&1 | tee $SCRATCH/warp-rocm-logs/mjw-details-$SLURM_JOB_ID.log | tail -15
echo "DETAILS_DONE"
