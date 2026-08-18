#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=robust-amd3
#SBATCH --output=/matx/u/knatalia/warp-rocm-logs/robust-amd3-%j.out
set -uxo pipefail
export SCRATCH=/matx/u/knatalia
export AGENT=$SCRATCH/agent-robust
export XDG_CACHE_HOME=$AGENT/.cache3-$SLURM_JOB_ID
export TMPDIR=$AGENT/.tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" $AGENT/ungated
cd $SCRATCH/warp-rocm && source .venv/bin/activate

echo "=============== PHASE 1: test_graph.py fully un-gated (incl. topology tests) ==============="
sed -e 's/^devices = \[d for d in get_test_devices() if not d.is_hip\]/devices = get_test_devices()/' \
    -e 's/^devices_with_cuda_graph_module_load = \[d for d in get_test_devices_with_cuda_graph_module_load() if not d.is_hip\]/devices_with_cuda_graph_module_load = get_test_devices_with_cuda_graph_module_load()/' \
    -e 's/^cuda_devices = \[d for d in get_selected_cuda_test_devices_with_mempool() if not d.is_hip\]/cuda_devices = get_selected_cuda_test_devices_with_mempool()/' \
    $SCRATCH/warp-rocm/warp/tests/test_graph.py > $AGENT/ungated/test_graph_ungated.py
grep -n "is_hip" $AGENT/ungated/test_graph_ungated.py || echo "NO_HIP_GATES_LEFT"
timeout 2400 python $AGENT/ungated/test_graph_ungated.py 2>&1 | tail -140
echo "PHASE1_END"

echo "=============== PHASE 2: null-stream torn-read stress ==============="
timeout 2400 python $AGENT/tools/null_stream_sync.py --spin 200000 --trials 10 --stress-iters 400
echo "PHASE2_END"

echo "=============== PHASE 3: suite-level flake hunt (15x TestAsync + TestFemField) ==============="
# The two intermittents were only ever seen in the full parallel suite runner,
# never standalone, so repeat them the way the suite runs them.
FAILS=0
for i in $(seq 1 15); do
  echo "--- flake run $i ---"
  timeout 1800 python -m warp.tests -s autodetect -k TestAsync -k TestFemField 2>&1 | tail -6 > $TMPDIR/flake-$SLURM_JOB_ID-$i.txt
  cat $TMPDIR/flake-$SLURM_JOB_ID-$i.txt
  grep -qE "^OK( \(|$)" $TMPDIR/flake-$SLURM_JOB_ID-$i.txt || { FAILS=$((FAILS+1)); echo "FLAKE_RUN_${i}_FAILED"; }
done
echo "FLAKE_HUNT_RESULT: $FAILS/15 runs failed"
echo "PHASE3_END"

echo "ROBUST_AMD3_DONE"
