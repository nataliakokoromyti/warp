#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=robust-amd2
#SBATCH --output=/matx/u/knatalia/warp-rocm-logs/robust-amd2-%j.out
set -uxo pipefail
export SCRATCH=/matx/u/knatalia
export AGENT=$SCRATCH/agent-robust
export XDG_CACHE_HOME=$AGENT/.cache2-$SLURM_JOB_ID
export TMPDIR=$AGENT/.tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd $SCRATCH/warp-rocm && source .venv/bin/activate

echo "=============== PHASE 1: null-stream implicit sync probe ==============="
python $AGENT/tools/null_stream_sync.py --spin 200000 --trials 25
echo "PHASE1_RC=$?"

echo "=============== PHASE 2: targeted copy repro (300 iters) ==============="
timeout 2400 python $AGENT/tools/copy_repro.py --variants target --iters 300
echo "PHASE2_RC=$?"

echo "=============== PHASE 3: copy variant sweep (20 iters each) ==============="
timeout 3600 python $AGENT/tools/copy_repro.py --variants all --iters 20
echo "PHASE3_RC=$?"

echo "=============== PHASE 4: un-gated test_graph.py on HIP ==============="
# test_graph.py hard-excludes HIP devices with a comment inherited from the era
# when HIP graph capture was disabled. rocm-117 enables capture, so run the
# suite un-gated to see whether the exclusion is hiding real coverage.
mkdir -p $AGENT/ungated
sed -e 's/^devices = \[d for d in get_test_devices() if not d.is_hip\]/devices = get_test_devices()/' \
    -e 's/^devices_with_cuda_graph_module_load = \[d for d in get_test_devices_with_cuda_graph_module_load() if not d.is_hip\]/devices_with_cuda_graph_module_load = get_test_devices_with_cuda_graph_module_load()/' \
    -e 's/^cuda_devices = \[d for d in get_selected_cuda_test_devices() if not d.is_hip\]/cuda_devices = get_selected_cuda_test_devices()/' \
    $SCRATCH/warp-rocm/warp/tests/test_graph.py > $AGENT/ungated/test_graph_ungated.py
diff <(grep -c is_hip $SCRATCH/warp-rocm/warp/tests/test_graph.py) <(grep -c is_hip $AGENT/ungated/test_graph_ungated.py) || true
grep -n "is_hip" $AGENT/ungated/test_graph_ungated.py || echo "NO_HIP_GATES_LEFT"
timeout 2400 python $AGENT/ungated/test_graph_ungated.py 2>&1 | tail -120
echo "PHASE4_RC=$?"

echo "=============== PHASE 5: un-gated test_ipc.py on HIP ==============="
sed -e 's/^non_hip_devices = \[d for d in cuda_devices if not d.is_hip\]/non_hip_devices = cuda_devices/' \
    $SCRATCH/warp-rocm/warp/tests/cuda/test_ipc.py > $AGENT/ungated/test_ipc_ungated.py
timeout 900 python $AGENT/ungated/test_ipc_ungated.py 2>&1 | tail -60
echo "PHASE5_RC=$?"

echo "ROBUST_AMD2_DONE"
