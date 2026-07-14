#!/bin/bash
# Launch TSG Benchmark Sweep — parallel across GPUs and environments
# Usage: bash scripts/launch_sweep.sh [--small] [--dataset heston] [--gpus 0,2,3]

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

SMALL=""
DATASETS="heston"
GPUS="0,2,3"
MAX_PARALLEL=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --small) SMALL="--small"; shift ;;
        --dataset) DATASETS="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --parallel) MAX_PARALLEL="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra GPU_ARR <<< "$GPUS"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "TSG Benchmark Sweep Launcher"
echo "  Date: $(date)"
echo "  Datasets: $DATASETS"
echo "  GPUs: $GPUS"
echo "  Parallel: $MAX_PARALLEL"
echo "  Small: ${SMALL:-no}"
echo "============================================"

# Define methods by environment
PT_METHODS="gtgan,fourierflows,csdi,diffusionts"  # skip tsdiff (known failure)
TF1_METHODS="timegan,rgan"
TVAE_METHODS="timevae"

# Temp directory for experiment tracking
TMPDIR="${PROJECT_DIR}/.sweep_tmp"
mkdir -p "$TMPDIR"

# Determine seq_lens for each dataset
heston_seq="24 64 128"
stocks_seq="24"
energy_seq="24"
sinusoidal_seq="24"

ENV_PATHS="--tf1 ${PROJECT_DIR}/miniconda3/envs/tf1_env/bin/python --pt ${PROJECT_DIR}/miniconda3/envs/common_pt/bin/python --tvae ${PROJECT_DIR}/miniconda3/envs/timevae_env/bin/python"

pids=()

# Function to run one experiment and track it
run_experiment() {
    local env_python="$1"
    local method="$2"
    local dataset="$3"
    local seq_len="$4"
    local seed="$5"
    local gpu_id="$6"
    local tag="${method}_${dataset}_s${seq_len}_seed${seed}"
    local logfile="${TMPDIR}/${tag}.log"

    CUDA_VISIBLE_DEVICES=$gpu_id \
    $env_python run_benchmark.py \
        --methods "$method" \
        --datasets "$dataset" \
        --seq-len $seq_len \
        --seed "$seed" \
        --gpus $gpu_id \
        --output "$RESULTS_DIR" \
        $SMALL \
        > "$logfile" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✅ $tag done (GPU $gpu_id)"
        # Extract loss info
        if grep -q "loss:" "$logfile"; then
            echo "    Last loss: $(grep "loss:" "$logfile" | tail -1)"
        fi
    else
        echo "[$(date '+%H:%M:%S')] ⚠ $tag FAILED (exit $exit_code, GPU $gpu_id)"
        tail -5 "$logfile" | sed 's/^/    /'
    fi
}

# Run all experiments for a given env/method across dataset/seq_len/seed
run_sweep() {
    local env_python="$1"
    local method="$2"
    local dataset="$3"
    local seeds="$4"
    shift 4

    # Get seq_lens for this dataset
    local seq_lens_var="${dataset}_seq"
    local seq_lens="${!seq_lens_var:-24}"

    for seq_len in $seq_lens; do
        for seed in $seeds; do
            # Wait for a free GPU slot
            while true; do
                # Count running experiments
                running=$(jobs -rp | wc -l)
                if [ "$running" -lt "$MAX_PARALLEL" ]; then
                    break
                fi
                sleep 10
            done

            # Pick GPU (round-robin)
            local gpu_idx=$(( (${#pids[@]}) % ${#GPU_ARR[@]} ))
            local gpu_id=${GPU_ARR[$gpu_idx]}

            # Launch in background
            run_experiment "$env_python" "$method" "$dataset" "$seq_len" "$seed" "$gpu_id" &
            pids+=($!)
        done
    done
}

# Always run on Heston first (priority)
for dataset in $(echo $DATASETS | tr ',' ' '); do
    echo ""
    echo "=== Dataset: $dataset ==="

    SEEDS="0 1 2 3 4"
    if [ -n "$SMALL" ]; then
        SEEDS="0"
    fi

    # Launch PyTorch methods
    echo "  Phase 1: PyTorch methods ($PT_METHODS)"
    IFS=',' read -ra PT_ARR <<< "$PT_METHODS"
    for method in "${PT_ARR[@]}"; do
        run_sweep "${PROJECT_DIR}/miniconda3/envs/common_pt/bin/python" "$method" "$dataset" "$SEEDS"
    done

    # Wait for all PT methods to finish
    echo "  Waiting for PyTorch methods to complete..."
    wait
    echo "  ✅ PyTorch methods done for $dataset"

    # Launch TF1 methods
    echo "  Phase 2: TF1 methods ($TF1_METHODS)"
    IFS=',' read -ra TF1_ARR <<< "$TF1_METHODS"
    for method in "${TF1_ARR[@]}"; do
        run_sweep "${PROJECT_DIR}/miniconda3/envs/tf1_env/bin/python" "$method" "$dataset" "$SEEDS"
    done

    wait
    echo "  ✅ TF1 methods done for $dataset"

    # Launch timeVAE
    echo "  Phase 3: timeVAE ($TVAE_METHODS)"
    IFS=',' read -ra TVAE_ARR <<< "$TVAE_METHODS"
    for method in "${TVAE_ARR[@]}"; do
        run_sweep "${PROJECT_DIR}/miniconda3/envs/timevae_env/bin/python" "$method" "$dataset" "$SEEDS"
    done

    wait
    echo "  ✅ timeVAE done for $dataset"

    echo "=== Dataset $dataset complete ==="
done

echo ""
echo "============================================"
echo "✅ ALL SWEEPS COMPLETE"
echo "  Finished: $(date)"
echo "============================================"

# Print summary
echo ""
echo "=== EXPERIMENT SUMMARY ==="
echo "Total experiments tracked: ${#pids[@]}"

# Check results
if [ -f "$RESULTS_DIR/all_results.jsonl" ]; then
    total=$(wc -l < "$RESULTS_DIR/all_results.jsonl")
    failed=$(grep -c '"FAILED"' "$RESULTS_DIR/all_results.jsonl" 2>/dev/null || echo 0)
    ok=$((total - failed))
    echo "  Total: $total experiments"
    echo "  OK: $ok"
    echo "  Failed: $failed"
fi
