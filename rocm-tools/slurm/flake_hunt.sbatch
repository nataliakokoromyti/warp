#!/bin/bash
# Full-suite flake hunt.
#
# The two known intermittents (test_copy_i2c_d2d_...Graph..., test_implicit_fields)
# have only ever appeared in *full* suite runs, where ~16 test classes share the GPU
# across parallel processes. Running the two classes alone does not reproduce them
# (0/300 and 0/500 targeted iterations), so reproduce the real conditions: repeat the
# whole suite and record every failure block.
#
# Observed base rate before this hunt: 2 failures across 9 recorded full-suite runs.
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --job-name=flake-hunt
#SBATCH --output=/matx/u/knatalia/warp-rocm-logs/flake-hunt-%j.out
set -uo pipefail
export SCRATCH=/matx/u/knatalia
export AGENT=$SCRATCH/agent-robust
export XDG_CACHE_HOME=$AGENT/.cache-flake-$SLURM_JOB_ID
export TMPDIR=$AGENT/.tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd $SCRATCH/warp-rocm && source .venv/bin/activate
python -c "import warp; print('warp', warp.__version__)"

RUNS=${FLAKE_RUNS:-14}
FAILS=0
for i in $(seq 1 "$RUNS"); do
  LOG=$TMPDIR/flakesuite-$SLURM_JOB_ID-$i.log
  echo "######## SUITE RUN $i/$RUNS ########"
  timeout 2400 python -m warp.tests -s autodetect > "$LOG" 2>&1
  tail -4 "$LOG"
  if grep -qE "^OK( \(|$)" "$LOG"; then
    echo "RUN_${i}_OK"
  else
    FAILS=$((FAILS + 1))
    echo "RUN_${i}_FAILED"
    echo "---- failure detail ----"
    grep -nE "^(FAIL|ERROR):" "$LOG"
    # print each failure block so the exact-value signature is captured
    awk '/^(FAIL|ERROR): /{p=1} p{print} /^-{60,}$/{if(p&&++n%2==0)p=0}' "$LOG" | head -120
    echo "---- end failure detail ----"
  fi
done

echo "FLAKE_HUNT_RESULT: $FAILS/$RUNS full-suite runs failed"
echo "FLAKE_HUNT_DONE"
