#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=col-valw
#SBATCH --output=/matx/u/knatalia/agent-collision/logs/valw-%j.out
# Warp test suite, with and without the HIP default-launch-bounds codegen change.
set -uxo pipefail
A=/matx/u/knatalia/agent-collision
export TMPDIR=$A/.tmp-valw-$SLURM_JOB_ID
mkdir -p $TMPDIR
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
source $A/lb/venv/bin/activate
cd $A/lb/warp-rocm
CG=$A/lb/warp-rocm/warp/_src/codegen.py

for V in on off; do
  if [ "$V" = on ]; then cp $A/lb/warp-rocm/warp/_src/codegen.new.py $CG
  else cp $A/lb/warp-rocm/warp/_src/codegen.py.orig $CG; fi
  echo "=================== WARP SUITE variant=$V ($(grep -c WP_DEFAULT_LAUNCH_BOUNDS $CG) macro hits)"
  export XDG_CACHE_HOME=$A/.cache-valw-$SLURM_JOB_ID-$V
  mkdir -p $XDG_CACHE_HOME
  python -m warp.tests -s autodetect > $A/logs/warpsuite-$SLURM_JOB_ID-$V.log 2>&1 || true
  grep -E "^Ran [0-9]+ tests|^OK|^FAILED|skipped=" $A/logs/warpsuite-$SLURM_JOB_ID-$V.log | tail -5
  echo "--- failures/errors detail (if any)"
  grep -E "^(FAIL|ERROR):" $A/logs/warpsuite-$SLURM_JOB_ID-$V.log | head -20 || true
done
cp $A/lb/warp-rocm/warp/_src/codegen.py.orig $CG
echo "VALW_DONE"
