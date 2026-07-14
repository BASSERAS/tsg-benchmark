#!/usr/bin/env python3
"""
TSG SOTA Benchmark — Unified runner.

Orchestrates: data generation → train 8 methods × 5 datasets × 3 seq_lens × 5 seeds → metrics → tables.

Usage:
    python run_benchmark.py                          # full sweep (expensive)
    python run_benchmark.py --dry-run --small        # 5% data, 1 seed, 1 method smoke test
    python run_benchmark.py --methods timegan,csdi    # subset of methods
    python run_benchmark.py --datasets sines,heston   # subset of datasets
    python run_benchmark.py --seq-len 24              # single seq_len
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure the project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Config ────────────────────────────────────────────────────────────────────

ALL_METHODS = [
    "timegan",
    "rgan",
    "gtgan",
    "timevae",
    "fourierflows",
    "csdi",
    "tsdiff",
    "diffusionts",
]

ALL_DATASETS = ["sines", "stocks", "energy", "heston", "sinusoidal"]

ALL_SEQ_LENS = [24, 64, 128]

SEEDS = [0, 1, 2, 3, 4]

# Fixed training budget (gradient steps) per seq_len
TRAINING_STEPS = {24: 5000, 64: 10000, 128: 15000}
BATCH_SIZE = 128

# ── Environment paths ─────────────────────────────────────────────────────────

CONDA_BASE = os.path.join(PROJECT_ROOT, "miniconda3")
ENV_PATHS = {
    "tf1": os.path.join(CONDA_BASE, "envs", "tf1_env", "bin", "python"),
    "timevae": os.path.join(CONDA_BASE, "envs", "timevae_env", "bin", "python"),
    "common_pt": os.path.join(CONDA_BASE, "envs", "common_pt", "bin", "python"),
}

# Which env each method needs
METHOD_ENV = {
    "timegan": "tf1",
    "rgan": "tf1",
    "gtgan": "common_pt",
    "timevae": "timevae",
    "fourierflows": "common_pt",
    "csdi": "common_pt",
    "tsdiff": "common_pt",
    "diffusionts": "common_pt",
}

# ── Data directory ─────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Import helpers ─────────────────────────────────────────────────────────────

def import_adapter(method: str):
    """Dynamically import an adapter class by method name."""
    if method == "timegan":
        from adapters.timegan_adapter import TimeGANAdapter
        return TimeGANAdapter
    elif method == "rgan":
        from adapters.rgan_adapter import RGANAdapter
        return RGANAdapter
    elif method == "gtgan":
        from adapters.gtgan_adapter import GTGANAdapter
        return GTGANAdapter
    elif method == "timevae":
        from adapters.timevae_adapter import TimeVAEAdapter
        return TimeVAEAdapter
    elif method == "fourierflows":
        from adapters.fourierflows_adapter import FourierFlowsAdapter
        return FourierFlowsAdapter
    elif method == "csdi":
        from adapters.csdi_adapter import CSDIAdapter
        return CSDIAdapter
    elif method == "tsdiff":
        from adapters.tsdiff_adapter import TSDiffAdapter
        return TSDiffAdapter
    elif method == "diffusionts":
        from adapters.diffusionts_adapter import DiffusionTSAdapter
        return DiffusionTSAdapter
    else:
        raise ValueError(f"Unknown method: {method}")


def load_dataset(name: str, seq_len: int, seed: int) -> dict:
    """Load or generate a dataset, returning standardized dict."""
    from data.datasets import (
        generate_sines,
        load_stock_data,
        load_energy_data,
        generate_heston,
        generate_sinusoidal_mixture,
        train_test_split,
        min_max_normalize,
    )

    heston_v = None
    if name == "sines":
        data = generate_sines(n_samples=10000, seq_len=seq_len, n_features=5, seed=seed)
    elif name == "stocks":
        data = load_stock_data(seq_len=seq_len)
    elif name == "energy":
        data = load_energy_data(seq_len=seq_len)
    elif name == "heston":
        S, v = generate_heston(n_samples=10000, seq_len=seq_len, seed=seed)
        data = S  # shape (N, seq_len, 1)
        heston_v = v  # shape (N, seq_len)
    elif name == "sinusoidal":
        data = generate_sinusoidal_mixture(
            n_samples=5000, seq_len=seq_len, K=3, snr_db=20.0, seed=seed
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Convert list of arrays to numpy if needed
    if isinstance(data, list):
        data = np.array(data)

    # Ensure 3D
    if data.ndim == 2:
        data = data.reshape(data.shape[0], data.shape[1], 1)

    train_data, test_data = train_test_split(data, train_ratio=0.8, seed=seed)
    train_norm, test_norm, min_vals, max_vals = min_max_normalize(train_data, test_data)

    result = {
        "train": train_norm,
        "test": test_norm,
        "train_raw": train_data,
        "test_raw": test_data,
        "min_vals": min_vals,
        "max_vals": max_vals,
        "n_features": data.shape[-1],
        "n_train": len(train_norm),
        "n_test": len(test_norm),
        "heston_v": heston_v,
    }
    return result


# ── GPU management ─────────────────────────────────────────────────────────────

def get_device(gpu_id: int = 0) -> str:
    """Get device string. Use CPU if no GPU available."""
    try:
        import torch
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            return f"cuda:{gpu_id}"
    except ImportError:
        pass
    return "cpu"


# ── Single experiment ──────────────────────────────────────────────────────────

def run_experiment(
    method: str,
    dataset_name: str,
    seq_len: int,
    seed: int,
    gpu_id: int = 0,
    small: bool = False,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Run one (method × dataset × seq_len × seed) experiment.
    Returns dict with metrics and metadata.
    """
    # Load data
    data_dict = load_dataset(dataset_name, seq_len, seed)

    train_data = data_dict["train"]
    test_data = data_dict["test"]

    # For small dry-run: use 5% of data
    if small:
        n_small = max(32, int(0.05 * len(train_data)))
        train_data = train_data[:n_small]

    device = get_device(gpu_id)
    n_features = data_dict["n_features"]

    # --- TRAIN ---
    print(f"\n{'='*60}")
    print(f"Training {method} on {dataset_name} (seq_len={seq_len}, seed={seed}, device={device})")
    print(f"  Train samples: {len(train_data)}, Features: {n_features}")
    print(f"{'='*60}")

    AdapterClass = import_adapter(method)
    adapter = AdapterClass(seq_len=seq_len, n_features=n_features, seed=seed, device=device)

    # Modify training steps for adapter
    if hasattr(adapter, 'training_steps'):
        adapter.training_steps = TRAINING_STEPS[seq_len]

    train_start = time.time()
    try:
        adapter.fit(train_data)
    except Exception as e:
        print(f"  ERROR: {method} failed during training: {e}")
        traceback.print_exc()
        return {"method": method, "dataset": dataset_name, "seq_len": seq_len,
                "seed": seed, "status": "FAILED", "error": str(e)}
    train_time = time.time() - train_start

    # --- SAMPLE ---
    n_samples = min(len(test_data), 2000)  # cap sampling for memory
    try:
        gen_data = adapter.sample(n_samples)
    except Exception as e:
        print(f"  ERROR: {method} failed during sampling: {e}")
        traceback.print_exc()
        return {"method": method, "dataset": dataset_name, "seq_len": seq_len,
                "seed": seed, "status": "FAILED", "error": str(e)}

    # Inverse transform generated data
    from data.datasets import inverse_min_max
    gen_data_orig = inverse_min_max(gen_data, data_dict["min_vals"], data_dict["max_vals"])

    # Save generated samples for A13/A14 post-processing from TF1
    _samp_dir = os.path.join(output_dir or RESULTS_DIR, "samples")
    os.makedirs(_samp_dir, exist_ok=True)
    _samp_path = os.path.join(_samp_dir, f"{method}_{dataset_name}_s{seq_len}_seed{seed}.npy")
    np.save(_samp_path, gen_data_orig.astype(np.float16))
    _real_path = os.path.join(_samp_dir, f"real_{dataset_name}_s{seq_len}.npy")
    if not os.path.exists(_real_path):
        np.save(_real_path, data_dict["test_raw"].astype(np.float16))

    # --- COMPUTE METRICS ---
    from metrics import (
        compute_all_metrics,
    )
    # Use real test data for comparison (same size as generated)
    real_compare = data_dict["test_raw"][:n_samples]

    try:
        metrics = compute_all_metrics(
            gen_data_orig, real_compare,
            heston_v=data_dict["heston_v"][:n_samples] if data_dict["heston_v"] is not None else None,
            seed=seed,
        )
    except Exception as e:
        print(f"  ERROR: metrics computation failed: {e}")
        traceback.print_exc()
        metrics = {"error": str(e)}

    # --- PARAMETER COUNT ---
    try:
        n_params = adapter.num_parameters()
    except:
        n_params = -1

    result = {
        "method": method,
        "dataset": dataset_name,
        "seq_len": seq_len,
        "seed": seed,
        "status": "OK",
        "train_time_sec": train_time,
        "peak_gpu_mem_mb": getattr(adapter, 'peak_gpu_mem_mb', 0.0),
        "num_parameters": n_params,
        **metrics,
    }

    print(f"  ✓ {method} done. Time: {train_time:.1f}s. Params: {n_params:,}")
    return result


# ── Main sweep ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TSG SOTA Benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Run with 5% data")
    parser.add_argument("--small", action="store_true", help="Alias for dry-run")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated method list")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset list")
    parser.add_argument("--seq-len", type=int, default=None, help="Single seq_len")
    parser.add_argument("--seed", type=str, default=None, help="Single seed (0), comma-separated (0,1,2), or 'all'")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="GPU IDs to use")
    parser.add_argument("--output", type=str, default=RESULTS_DIR, help="Results directory")
    args = parser.parse_args()

    is_small = args.dry_run or args.small

    methods = args.methods.split(",") if args.methods else ALL_METHODS
    datasets = args.datasets.split(",") if args.datasets else ALL_DATASETS
    seq_lens = [args.seq_len] if args.seq_len else ALL_SEQ_LENS
    seeds = [0] if is_small else SEEDS
    if args.seed is not None:
        if args.seed == "all":
            seeds = SEEDS
        else:
            seeds = [int(s) for s in args.seed.split(",")]
    gpu_ids = [int(g) for g in args.gpus.split(",")]

    if is_small:
        print("🔬 SMALL MODE: 5% data, 1 seed, 1 method per GPU (smoke test)")
    else:
        print(f"🚀 FULL SWEEP: {len(methods)} methods × {len(datasets)} datasets × "
              f"{len(seq_lens)} seq_lens × {len(seeds)} seeds")

    print(f"Methods: {methods}")
    print(f"Datasets: {datasets}")
    print(f"Seq lens: {seq_lens}")
    print(f"Seeds: {seeds}")
    print(f"GPUs: {gpu_ids}")
    print()

    all_results = []

    for method in methods:
        for dataset_name in datasets:
            for seq_len in seq_lens:
                for seed_idx, seed in enumerate(seeds):
                    gpu_id = gpu_ids[(len(all_results)) % len(gpu_ids)]
                    result = run_experiment(
                        method=method,
                        dataset_name=dataset_name,
                        seq_len=seq_len,
                        seed=seed,
                        gpu_id=gpu_id,
                        small=is_small,
                        output_dir=args.output,
                    )
                    all_results.append(result)

                    # Save intermediate results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    with open(os.path.join(args.output, "all_results.jsonl"), "a") as f:
                        f.write(json.dumps(result) + "\n")

    # Compute aggregates
    if not is_small:
        from metrics import compute_aggregate_tables
        compute_aggregate_tables(all_results, output_dir=args.output)
        print("\n✅ Tables saved to", args.output)

    print(f"\nDone. {len(all_results)} experiments completed.")
    return all_results


if __name__ == "__main__":
    main()
