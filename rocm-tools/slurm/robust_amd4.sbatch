#!/bin/bash
# Enumerate exactly which of test_graph.py's HIP-excluded tests pass, fail, or
# crash the GPU. One process per test: the first crash (HSA memory access fault)
# otherwise aborts the run and hides everything after it.
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=robust-amd4
#SBATCH --output=/matx/u/knatalia/warp-rocm-logs/robust-amd4-%j.out
set -uo pipefail
export SCRATCH=/matx/u/knatalia
export AGENT=$SCRATCH/agent-robust
export XDG_CACHE_HOME=$AGENT/.cache4-$SLURM_JOB_ID
export TMPDIR=$AGENT/.tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" $AGENT/ungated
cd $SCRATCH/warp-rocm && source .venv/bin/activate

sed -e 's/^devices = \[d for d in get_test_devices() if not d.is_hip\]/devices = get_test_devices()/' \
    -e 's/^devices_with_cuda_graph_module_load = \[d for d in get_test_devices_with_cuda_graph_module_load() if not d.is_hip\]/devices_with_cuda_graph_module_load = get_test_devices_with_cuda_graph_module_load()/' \
    -e 's/^cuda_devices = \[d for d in get_selected_cuda_test_devices_with_mempool() if not d.is_hip\]/cuda_devices = get_selected_cuda_test_devices_with_mempool()/' \
    $SCRATCH/warp-rocm/warp/tests/test_graph.py > $AGENT/ungated/test_graph_ungated.py
grep -n "is_hip" $AGENT/ungated/test_graph_ungated.py || echo "NO_HIP_GATES_LEFT"

echo "=============== warm the kernel cache (one pass, crashes tolerated) ==============="
timeout 900 python $AGENT/ungated/test_graph_ungated.py > /dev/null 2>&1 || true

echo "=============== isolated per-test sweep ==============="
timeout 7200 python $AGENT/tools/isolate_tests.py $AGENT/ungated/test_graph_ungated.py --filter cuda_0
echo "SWEEP_END"

echo "ROBUST_AMD4_DONE"
