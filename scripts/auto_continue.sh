#!/bin/bash
# Continuation script: runs after Heston seq24 completes
# Launches Heston seq64, seq128, Stocks, Energy, Sinusoidal

PROJECT_DIR="/home/tbasseras/tsg-benchmark"
cd "$PROJECT_DIR"

# Paths
PT_PY="$PROJECT_DIR/miniconda3/envs/common_pt/bin/python"
TF1_PY="$PROJECT_DIR/miniconda3/envs/tf1_env/bin/python"
TVAE_PY="$PROJECT_DIR/miniconda3/envs/timevae_env/bin/python"

echo "========================================"
echo "TSG Benchmark Continuation"
echo "Started: $(date)"
echo "========================================"

# Run a group of methods in parallel across GPUs
# Uses direct background + poll-wait (no nohup, so wait works)
run_parallel() {
    local dataset="$1"
    local seq_len="$2"
    shift 2

    local gpu=0
    local pids=""

    for method in "$@"; do
        case $method in
            timegan|rgan) python=$TF1_PY ;;
            timevae) python=$TVAE_PY ;;
            *) python=$PT_PY ;;
        esac

        local log=".sweep_${method}_${dataset}_s${seq_len}.log"
        echo "[$(date '+%H:%M:%S')] GPU$gpu: $method $dataset s${seq_len}"

        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $python -u run_benchmark.py \
            --methods "$method" --datasets "$dataset" --seq-len $seq_len --seed all --gpus 0 \
            > "$log" 2>&1 &
        local child_pid=$!
        pids="$pids $child_pid"
        gpu=$(( (gpu + 1) % 4 ))
    done

    # Poll-wait for all PIDs to finish
    for pid in $pids; do
        while kill -0 $pid 2>/dev/null; do
            sleep 30
        done
        echo "  PID $pid finished"
    done
}

# Heston seq64
echo "=== Heston seq_len=64 ==="
run_parallel "heston" 64 "gtgan" "fourierflows" "diffusionts" "csdi"
run_parallel "heston" 64 "timegan" "rgan" "timevae"

# Heston seq128
echo "=== Heston seq_len=128 ==="
run_parallel "heston" 128 "gtgan" "fourierflows" "diffusionts" "csdi"
run_parallel "heston" 128 "timegan" "rgan" "timevae"

# Stocks
echo "=== Stocks (GOOG) ==="
run_parallel "stocks" 24 "gtgan" "fourierflows" "diffusionts" "csdi"
run_parallel "stocks" 24 "timegan" "rgan" "timevae"

# Energy
echo "=== Energy (UCI) ==="
run_parallel "energy" 24 "gtgan" "fourierflows" "diffusionts" "csdi"
run_parallel "energy" 24 "timegan" "rgan" "timevae"

# Sinusoidal
echo "=== Sinusoidal Mixture ==="
run_parallel "sinusoidal" 24 "gtgan" "fourierflows" "diffusionts" "csdi"
run_parallel "sinusoidal" 24 "timegan" "rgan" "timevae"

echo ""
echo "========================================"
echo "ALL SWEEPS COMPLETE: $(date)"
echo "========================================"

# Generate tables and push
$PT_PY scripts/generate_tables.py
cd "$PROJECT_DIR"
git add results/
git commit -m "Auto: full benchmark sweep complete $(date +%Y-%m-%d)"
git push 2>&1 || echo "Push failed (non-fast-forward), pulling first..." && git pull --rebase && git push

echo "Done!"
