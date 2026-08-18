#!/bin/bash
# Robustness probe battery: everything used to characterize the intermittent suite
# failures, the primitives launch failure, and the HIP test gates. Runs on MI350X by
# default; set PROBES_NV=1 to run the vendor-control subset on an L40S instead
# (submit with --gres=gpu:l40s:1 and no --nodelist in that case).
#
#   sbatch rocm-tools/slurm/robust_probes.sbatch
#
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodelist=matx-amd-1
#SBATCH --gres=gpu:mi350x:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=robust-probes
#SBATCH --output=/matx/u/knatalia/warp-rocm-logs/robust-probes-%j.out
set -uo pipefail
export SCRATCH=${SCRATCH:-/matx/u/knatalia}
export WARP_DIR=${WARP_DIR:-$SCRATCH/warp-rocm}
export TOOLS=${TOOLS:-$WARP_DIR/rocm-tools}
# Per-job kernel cache: concurrent jobs sharing one Warp cache corrupt each other.
export XDG_CACHE_HOME=$SCRATCH/.cache-probes-${SLURM_JOB_ID:-manual}
export TMPDIR=$SCRATCH/.tmp
export UNGATED=$TMPDIR/ungated-${SLURM_JOB_ID:-manual}
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$UNGATED"
cd "$WARP_DIR" && source .venv/bin/activate
python -c "import warp; print('warp', warp.__version__)"

run() { echo; echo "############ $1 ############"; shift; "$@"; echo "rc=$?"; }

# --- launch limits -----------------------------------------------------------
run "oversized launches (HSA uint32 dispatch ceiling)" \
  timeout 900 python "$TOOLS/big_launch.py"

# --- torn device-to-host reads (the intermittent suite failures) --------------
run "null-stream ordering + torn-read stress" \
  timeout 2400 python "$TOOLS/null_stream_sync.py" --spin 200000 --trials 25 --stress-iters 400
run "captured cross-stream fork/join ordering" \
  timeout 900 python "$TOOLS/capture_fork_join.py" --spin 200000 --trials 20
run "the exact intermittent copy configuration, 300x" \
  timeout 2400 python "$TOOLS/copy_repro.py" --variants target --iters 300
run "all non-contiguous d2d copy variants" \
  timeout 3600 python "$TOOLS/copy_repro.py" --variants all --iters 20
run "fem implicit fields x500, then x300 under GPU load" \
  timeout 1800 python "$TOOLS/flake_hunt.py" --test fem_implicit --iters 500
timeout 1800 python "$TOOLS/flake_hunt.py" --test fem_implicit --iters 300 --load 3

# --- graph allocation fault --------------------------------------------------
run "in-capture allocation fault decomposition" \
  timeout 4000 python "$TOOLS/graph_alloc_fault.py"

# --- gate audit: what do the HIP exclusions actually hide? --------------------
# test_graph.py and test_ipc.py filter HIP out with hard `not d.is_hip` lists.
sed -e 's/^devices = \[d for d in get_test_devices() if not d.is_hip\]/devices = get_test_devices()/' \
    -e 's/^devices_with_cuda_graph_module_load = \[d for d in get_test_devices_with_cuda_graph_module_load() if not d.is_hip\]/devices_with_cuda_graph_module_load = get_test_devices_with_cuda_graph_module_load()/' \
    -e 's/^cuda_devices = \[d for d in get_selected_cuda_test_devices_with_mempool() if not d.is_hip\]/cuda_devices = get_selected_cuda_test_devices_with_mempool()/' \
    "$WARP_DIR/warp/tests/test_graph.py" > "$UNGATED/test_graph_ungated.py"
sed -e 's/^non_hip_devices = \[d for d in cuda_devices if not d.is_hip\]/non_hip_devices = cuda_devices/' \
    "$WARP_DIR/warp/tests/cuda/test_ipc.py" > "$UNGATED/test_ipc_ungated.py"
grep -c is_hip "$UNGATED/test_graph_ungated.py" "$UNGATED/test_ipc_ungated.py" || true

# One process per test: a GPU fault kills the process and would hide every later test.
run "un-gated test_graph.py, isolated per test" \
  timeout 7200 python "$TOOLS/isolate_tests.py" "$UNGATED/test_graph_ungated.py" --filter cuda_0
run "un-gated test_ipc.py, isolated per test" \
  timeout 1800 python "$TOOLS/isolate_tests.py" "$UNGATED/test_ipc_ungated.py" --filter cuda_0

echo "ROBUST_PROBES_DONE"
