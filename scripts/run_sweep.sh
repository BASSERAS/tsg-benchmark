#!/bin/bash
# TSG Benchmark — Run Full Sweep
# Usage: bash scripts/run_sweep.sh [options]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BENCH_DIR"

echo "TSG SOTA Benchmark"
echo "=================="
echo ""

# Default: run everything
ARGS="$@"

if [ -z "$ARGS" ]; then
    echo "Starting full 5-seed sweep..."
    echo "This will take several hours."
    echo ""
    # Small validation first
    echo "Step 1: Validation (Sines seq_len=24, 1 seed)"
    ./miniconda3/envs/common_pt/bin/python run_benchmark.py --small --datasets sines --methods timegan,rgan,csdi,fourierflows --seq-len 24
    echo ""
    # Full sweep
    echo "Step 2: Full sweep"
    ./miniconda3/envs/common_pt/bin/python run_benchmark.py
else
    ./miniconda3/envs/common_pt/bin/python run_benchmark.py $ARGS
fi
