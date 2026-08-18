#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=stepper-amd
#SBATCH --output=/matx/u/knatalia/agent-stepper/logs/amd-%j.out
set -uxo pipefail
export SCRATCH=/matx/u/knatalia
export AS=$SCRATCH/agent-stepper
export XDG_CACHE_HOME=$AS/.cache-amd-$SLURM_JOB_ID
export TMPDIR=/tmp
cd $SCRATCH/warp-rocm && source .venv/bin/activate
export PYTHONPATH=$AS/mjw-amd
python -c "import mujoco_warp,warp;print('mjw from',mujoco_warp.__file__);print('warp',warp.config.version)"

echo "=== scene validation, auto policy (no conditional nodes: expect the split) ==="
for i in 0 1 2 3 4 5 6; do
  python $AS/stepper_validate.py $SCRATCH/.bench_assets $i auto 2>&1 | grep -E "^STEP|^===|Error|Traceback" || true
done

echo "=== end-to-end testspeed A/B (cli.unroll picks up the Stepper automatically) ==="
cd $AS/mjw-amd
for args in "franka_emika_panda/scene.xml --nworld=32768 --nconmax=1 --njmax=5" \
            "humanoid/humanoid.xml --nworld=8192 --nconmax=24 --njmax=64" \
            "unitree_g1_flat/scene_flat.xml --nworld=8192 --nconmax=48 --njmax=192"; do
  set -- $args
  mjcf=$1; shift
  for ck in true false; do
    echo "TESTSPEED $mjcf chunked_solver=$ck"
    python -m mujoco_warp.testspeed $SCRATCH/.bench_assets/$mjcf "$@" \
      --format=human --nstep=1000 --event_trace=false --measure_solver=true \
      --chunked_solver=$ck 2>&1 | grep -E "chunked solver|steps per second|Summary for" || true
  done
done

echo "=== unit tests ==="
cd $AS/mjw-amd
python -m pytest -q --tb=short mujoco_warp/_src/stepper_test.py 2>&1 | grep -v "^Module \|HIPRTC" | tail -60
python -m pytest -q mujoco_warp/_src/solver_test.py mujoco_warp/_src/sleep_test.py \
  mujoco_warp/_src/forward_test.py mujoco_warp/_src/island_test.py 2>&1 | grep -v "^Module \|HIPRTC" | tail -20
echo "STEPPER_AMD_DONE"
