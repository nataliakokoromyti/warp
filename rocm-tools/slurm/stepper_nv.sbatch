#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=stepper-nv
#SBATCH --output=/matx/u/knatalia/agent-stepper/logs/nv-%j.out
set -uxo pipefail
export SCRATCH=/matx/u/knatalia
export AS=$SCRATCH/agent-stepper
export XDG_CACHE_HOME=$AS/.cache-nv-$SLURM_JOB_ID
export TMPDIR=/tmp
source $SCRATCH/nvidia-baseline/venv/bin/activate
export PYTHONPATH=$AS/mjw-nv
nvidia-smi --query-gpu=name --format=csv,noheader
python -c "import mujoco_warp;print('mjw from',mujoco_warp.__file__)"

echo "=== unit tests (stepper + the modules the solver refactor touches) ==="
cd $AS/mjw-nv
python -m pytest -x -q mujoco_warp/_src/stepper_test.py 2>&1 | tail -30
python -m pytest -q mujoco_warp/_src/solver_test.py mujoco_warp/_src/sleep_test.py \
  mujoco_warp/_src/forward_test.py mujoco_warp/_src/island_test.py 2>&1 | tail -20

echo "=== scene validation, auto policy (conditional nodes present: expect no split) ==="
for i in 0 1 2 3 4 5 6; do
  python $AS/stepper_validate.py $SCRATCH/.bench_assets $i auto 2>&1 | grep -E "^STEP|^===|Error|Traceback" || true
done

echo "=== scene validation, split forced (validates the decomposition itself) ==="
export STEPPER_FORCE=1
for i in 0 1 2 3 4 5 6; do
  python $AS/stepper_validate.py $SCRATCH/.bench_assets $i auto 2>&1 | grep -E "^STEP|^===|Error|Traceback" || true
done
echo "STEPPER_NV_DONE"
