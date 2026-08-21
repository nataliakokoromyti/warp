#!/bin/bash
# NVIDIA control for the robustness probes: the same scripts against stock upstream
# Warp on an L40S, so any AMD-only result can be shown to be AMD-only.
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=robust-nv
#SBATCH --output=/matx/u/knatalia/warp-rocm-logs/robust-nv-%j.out
set -uo pipefail
export SCRATCH=${SCRATCH:-/matx/u/knatalia}
export WARP_DIR=${WARP_DIR:-$SCRATCH/warp-rocm}
export TOOLS=${TOOLS:-$WARP_DIR/rocm-tools}
export XDG_CACHE_HOME=$SCRATCH/.cache-nv-${SLURM_JOB_ID:-manual}
export TMPDIR=$SCRATCH/.tmp
export UNGATED=$TMPDIR/ungated-nv-${SLURM_JOB_ID:-manual}
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$UNGATED"
# Stock upstream Warp, no ROCm patches.
source "${NV_VENV:-$SCRATCH/nvidia-baseline/venv}/bin/activate"
python -c "import warp; print('warp', warp.__version__)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

run() { echo; echo "############ $1 ############"; shift; "$@"; echo "rc=$?"; }

run "oversized launches (no ceiling expected on CUDA)" \
  timeout 900 python "$TOOLS/big_launch.py"
run "null-stream ordering + torn-read stress" \
  timeout 2400 python "$TOOLS/null_stream_sync.py" --spin 200000 --trials 25 --stress-iters 400
run "captured cross-stream fork/join ordering" \
  timeout 900 python "$TOOLS/capture_fork_join.py" --spin 200000 --trials 20
run "the exact intermittent copy configuration, 300x" \
  timeout 2400 python "$TOOLS/copy_repro.py" --variants target --iters 300
run "all non-contiguous d2d copy variants" \
  timeout 3600 python "$TOOLS/copy_repro.py" --variants all --iters 20
run "in-capture allocation fault decomposition" \
  timeout 4000 python "$TOOLS/graph_alloc_fault.py"

# The graph suite un-gated, to show the tests HIP skips are all expected to pass.
sed -e 's/^devices = \[d for d in get_test_devices() if not d.is_hip\]/devices = get_test_devices()/' \
    -e 's/^devices_with_cuda_graph_module_load = \[d for d in get_test_devices_with_cuda_graph_module_load() if not d.is_hip\]/devices_with_cuda_graph_module_load = get_test_devices_with_cuda_graph_module_load()/' \
    -e 's/^cuda_devices = \[d for d in get_selected_cuda_test_devices_with_mempool() if not d.is_hip\]/cuda_devices = get_selected_cuda_test_devices_with_mempool()/' \
    "$WARP_DIR/warp/tests/test_graph.py" > "$UNGATED/test_graph_nv.py"
run "un-gated test_graph.py, isolated per test" \
  timeout 3600 python "$TOOLS/isolate_tests.py" "$UNGATED/test_graph_nv.py" --filter cuda_0

echo "ROBUST_NV_DONE"
