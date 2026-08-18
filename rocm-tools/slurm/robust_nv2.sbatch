#!/bin/bash
# NVIDIA control for the graph-allocation fault found on MI350X.
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=robust-nv2
#SBATCH --output=/matx/u/knatalia/warp-rocm-logs/robust-nv2-%j.out
set -uo pipefail
export SCRATCH=/matx/u/knatalia
export AGENT=$SCRATCH/agent-robust
export XDG_CACHE_HOME=$AGENT/.cache-nv2-$SLURM_JOB_ID
export TMPDIR=$AGENT/.tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
source $SCRATCH/nvidia-baseline/venv/bin/activate
python -c "import warp; print('warp', warp.__version__)"
nvidia-smi --query-gpu=name --format=csv,noheader

# The rocm-117 tree's test_graph.py runs against stock Warp: it only imports
# warp.tests.unittest_utils, which the installed wheel provides.
cp $SCRATCH/warp-rocm/warp/tests/test_graph.py $AGENT/ungated/test_graph_nv.py
sed -i -e 's/^devices = \[d for d in get_test_devices() if not d.is_hip\]/devices = get_test_devices()/' \
       -e 's/^devices_with_cuda_graph_module_load = \[d for d in get_test_devices_with_cuda_graph_module_load() if not d.is_hip\]/devices_with_cuda_graph_module_load = get_test_devices_with_cuda_graph_module_load()/' \
       -e 's/^cuda_devices = \[d for d in get_selected_cuda_test_devices_with_mempool() if not d.is_hip\]/cuda_devices = get_selected_cuda_test_devices_with_mempool()/' \
       $AGENT/ungated/test_graph_nv.py

echo "=============== the specific failing test on CUDA ==============="
timeout 900 python $AGENT/ungated/test_graph_nv.py TestGraph.test_cuda_graph_alloc_transient_stream_cuda_0 2>&1 | tail -20
echo "=============== isolated sweep on CUDA ==============="
timeout 3000 python $AGENT/tools/isolate_tests.py $AGENT/ungated/test_graph_nv.py --filter cuda_0 2>&1 | tail -60
echo "ROBUST_NV2_DONE"
