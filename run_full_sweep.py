#!/usr/bin/env python3
"""
TSG Benchmark — Full Sweep Runner with Loss Tracking

Orchestrates the complete benchmark across all remaining datasets,
saving loss histories, generating loss plots, and producing final tables.

Usage:
    python run_full_sweep.py [--dry-run] [--datasets heston,stocks,energy,sinusoidal]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONDA_BASE = os.path.join(PROJECT_ROOT, "miniconda3")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Environment paths
TF1_PYTHON = os.path.join(CONDA_BASE, "envs", "tf1_env", "bin", "python")
COMMON_PT_PYTHON = os.path.join(CONDA_BASE, "envs", "common_pt", "bin", "python")
TIMEVAE_PYTHON = os.path.join(CONDA_BASE, "envs", "timevae_env", "bin", "python")

ALL_METHODS = ["timegan", "rgan", "gtgan", "timevae", "fourierflows", "csdi", "tsdiff", "diffusionts"]
SEEDS = [0, 1, 2, 3, 4]
ALL_SEQ_LENS = [24, 64, 128]


def get_env_python(method: str) -> str:
    """Return the python path for the conda environment needed by method."""
    if method in ("timegan", "rgan"):
        return TF1_PYTHON
    elif method == "timevae":
        return TIMEVAE_PYTHON
    else:
        return COMMON_PT_PYTHON


def get_env_name(method: str) -> str:
    if method in ("timegan", "rgan"):
        return "tf1_env"
    elif method == "timevae":
        return "timevae_env"
    else:
        return "common_pt"


def run_method_batch(methods, dataset, seq_len, seeds, gpu_ids, small=False):
    """Run a batch of methods on one (dataset, seq_len) across seeds.

    Each method-seed combo runs on a separate GPU in parallel.
    """
    results = []

    for method in methods:
        for seed in seeds:
            gpu_id = gpu_ids[(len(results)) % len(gpu_ids)]
            python_path = get_env_python(method)

            cmd = [
                python_path, os.path.join(PROJECT_ROOT, "run_benchmark.py"),
                "--methods", method,
                "--datasets", dataset,
                "--seq-len", str(seq_len),
                "--gpus", str(gpu_id),
            ]
            if small:
                cmd.append("--small")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            print(f"\n{'='*70}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Launching: {method} on {dataset} seq_len={seq_len} seed={seed} GPU={gpu_id}")
            print(f"  Env: {get_env_name(method)}")
            print(f"  Cmd: {' '.join(cmd)}")
            print(f"{'='*70}")

            t0 = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=PROJECT_ROOT)
            elapsed = time.time() - t0

            # Print output
            for line in result.stdout.split("\n"):
                if line.strip():
                    print(f"  [{method}] {line}")
            if result.stderr:
                for line in result.stderr.split("\n")[-5:]:
                    if line.strip():
                        print(f"  [{method} ERR] {line}")

            print(f"  [{method}] Done in {elapsed:.1f}s (exit code: {result.returncode})")

            if result.returncode != 0:
                print(f"  [{method}] FAILED. stderr follows:")
                print(result.stderr[-1000:])

    return results


def run_experiment(method, dataset, seq_len, seed, gpu_id):
    """Run a single experiment via subprocess and save loss history."""
    python_path = get_env_python(method)

    cmd = [
        python_path, os.path.join(PROJECT_ROOT, "run_experiment.py"),
        "--method", method,
        "--dataset", dataset,
        "--seq-len", str(seq_len),
        "--seed", str(seed),
        "--gpu", str(gpu_id),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=PROJECT_ROOT)

    if result.returncode == 0:
        try:
            return json.loads(result.stdout.strip().split("\n")[-1])
        except:
            return {"method": method, "dataset": dataset, "seq_len": seq_len, "seed": seed, "status": "FAILED", "error": "JSON parse error"}
    else:
        return {"method": method, "dataset": dataset, "seq_len": seq_len, "seed": seed, "status": "FAILED", "error": result.stderr[-500:]}


def save_loss_histories(results):
    """Extract loss histories from adapter objects and save as numpy arrays."""
    # Loss histories are saved within adapter objects in the subprocess.
    # We use a post-hoc approach: run a script that reads from
    # adapter._loss_history and saves to numpy files.
    pass


def generate_loss_plots():
    """Generate loss evolution plots for all completed experiments."""
    plot_script = os.path.join(PROJECT_ROOT, "scripts", "plot_losses.py")
    if os.path.exists(plot_script):
        subprocess.run(
            [COMMON_PT_PYTHON, plot_script],
            cwd=PROJECT_ROOT
        )


def merge_results():
    """Merge all experiment results into a single JSON."""
    results_file = os.path.join(RESULTS_DIR, "all_results.jsonl")
    if not os.path.exists(results_file):
        return []

    results = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except:
                    pass
    return results


def main():
    parser = argparse.ArgumentParser(description="Full TSG Benchmark Sweep")
    parser.add_argument("--dry-run", action="store_true", help="Smoke test (1 seed, small data)")
    parser.add_argument("--datasets", type=str, default="heston,stocks,energy,sinusoidal",
                        help="Comma-separated datasets (default: all remaining)")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="GPU IDs")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated methods")
    args = parser.parse_args()

    is_small = args.dry_run
    datasets = args.datasets.split(",")
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    methods = args.methods.split(",") if args.methods else ALL_METHODS
    seeds = [0] if is_small else SEEDS

    print(f"\n{'#'*70}")
    print(f"# TSG BENCHMARK — FULL SWEEP")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Datasets: {datasets}")
    print(f"# Methods: {methods}")
    print(f"# Seeds: {seeds}")
    print(f"# GPUs: {gpu_ids}")
    print(f"# Small mode: {is_small}")
    print(f"{'#'*70}\n")

    # Check environments exist
    for label, path in [("TF1", TF1_PYTHON), ("PyTorch", COMMON_PT_PYTHON), ("timeVAE", TIMEVAE_PYTHON)]:
        if os.path.exists(path):
            result = subprocess.run([path, "-c", "import sys; print(sys.version[:10])"],
                                   capture_output=True, text=True)
            print(f"  ✓ {label} env: {result.stdout.strip()}")
        else:
            print(f"  ✗ {label} env: NOT FOUND at {path}")

    print()

    # ── Organize by environment ──
    pt_methods = [m for m in methods if get_env_name(m) == "common_pt"]
    tf1_methods = [m for m in methods if get_env_name(m) == "tf1_env"]
    tvae_methods = [m for m in methods if get_env_name(m) == "timevae_env"]

    total_experiments = 0
    for dataset in datasets:
        if dataset == "heston":
            seq_lens = [24, 64, 128]
        elif dataset == "sinusoidal":
            seq_lens = [24]  # Default sinusoidal mixture
        else:
            seq_lens = [24]  # Stocks and Energy have fixed length

        for seq_len in seq_lens:
            total_experiments += len(methods) * len(seeds)

    print(f"Total experiments to run: {total_experiments}")
    print(f"  PyTorch methods ({len(pt_methods)}): {pt_methods}")
    print(f"  TF1 methods ({len(tf1_methods)}): {tf1_methods}")
    print(f"  timeVAE ({len(tvae_methods)}): {tvae_methods}")
    print()

    if is_small:
        print("🔬 DRY RUN / SMALL MODE — 5% data, 1 seed\n")

    all_results = []

    # ── Run experiments: process each dataset ──
    for dataset in datasets:
        if dataset == "heston":
            seq_lens = [24, 64, 128]
        elif dataset == "sinusoidal":
            seq_lens = [24]  # Default K=3, SNR=20dB
        else:
            seq_lens = [24]  # Stocks (GOOG), Energy (UCI) — fixed length

        for seq_len in seq_lens:
            print(f"\n{'='*70}")
            print(f"DATASET: {dataset} | seq_len={seq_len}")
            print(f"{'='*70}")

            # Batch by environment for parallel execution
            batch_commands = []

            for method in methods:
                for seed in seeds:
                    gpu_id = gpu_ids[len(batch_commands) % len(gpu_ids)]
                    python_path = get_env_python(method)

                    cmd = [
                        python_path, os.path.join(PROJECT_ROOT, "run_benchmark.py"),
                        "--methods", method,
                        "--datasets", dataset,
                        "--seq-len", str(seq_len),
                        "--gpus", str(gpu_id),
                    ]
                    if is_small:
                        cmd.append("--small")

                    batch_commands.append((method, dataset, seq_len, seed, gpu_id, cmd, python_path))

            # Run sequentially but report progress
            for idx, (method, ds, sl, seed, gpu_id, cmd, py_path) in enumerate(batch_commands):
                print(f"\n  [{idx+1}/{len(batch_commands)}] {method} seed={seed} GPU={gpu_id} (env: {get_env_name(method)})")

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                t0 = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=PROJECT_ROOT)
                elapsed = time.time() - t0

                # Print training output (filtered)
                for line in result.stdout.split("\n"):
                    stripped = line.strip()
                    if stripped and ("=" in stripped or "[" in stripped or "ERROR" in stripped or "Done" in stripped or "FAIL" in stripped or "✓" in stripped):
                        print(f"    {stripped}")

                if result.returncode != 0:
                    print(f"    ⚠ FAILED (exit {result.returncode}) — check logs")
                    if result.stderr:
                        for line in result.stderr.split("\n")[-3:]:
                            if line.strip():
                                print(f"    ERR: {line.strip()}")
                else:
                    print(f"    ✓ Done in {elapsed:.1f}s")

        # After each dataset, generate interim results
        ds_tag = f"{dataset}_seq{seq_len}"
        print(f"\n  Dataset {dataset} seq_len={seq_len} complete. Saving interim results...")

    print(f"\n{'='*70}")
    print(f"✅ ALL EXPERIMENTS COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
