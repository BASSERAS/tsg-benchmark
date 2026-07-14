#!/usr/bin/env python3
"""
Save loss histories from adapter objects to disk.

This script is called after benchmark runs to persist in-memory
loss histories to numpy/JSON files for plotting.

Usage:
    python scripts/save_loss_histories.py [--results-dir results]
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_loss_history(adapter, method, dataset, seq_len, seed, output_dir):
    """Save loss history from adapter object to disk."""
    if not hasattr(adapter, '_loss_history') or not adapter._loss_history:
        return

    hist = adapter._loss_history
    # Filter out empty histories
    hist = {k: v for k, v in hist.items() if v}
    if not hist:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    json_path = os.path.join(output_dir, f"{method}_seed{seed}_loss_history.json")
    data = {
        "method": method,
        "dataset": dataset,
        "seq_len": seq_len,
        "seed": seed,
        "loss_history": {k: [[int(s), float(v)] for s, v in vals] for k, vals in hist.items()},
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # Save as numpy arrays too
    npz_path = os.path.join(output_dir, f"{method}_seed{seed}_loss.npz")
    np.savez(npz_path, **{k: np.array(vals) for k, vals in hist.items()})

    return json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    args = parser.parse_args()

    # Find all experiment result files and look for saved loss histories
    print(f"Scanning {args.results_dir} for loss histories...")

    # Check for adapter loss history saves from runner
    jsonl_path = os.path.join(args.results_dir, "loss_histories.jsonl")
    if os.path.exists(jsonl_path):
        print(f"Found {jsonl_path}")
        # Group by experiment
        from collections import defaultdict
        experiments = defaultdict(dict)
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    key = (entry["method"], entry["dataset"], entry["seq_len"], entry["seed"])
                    experiments[key] = entry

        print(f"  {len(experiments)} experiments found")

        for (method, dataset, seq_len, seed), entry in experiments.items():
            loss_hist = entry.get("loss_history", {})
            if not loss_hist:
                continue
            output_dir = os.path.join(args.results_dir, f"{dataset}_seq{seq_len}")
            path = save_loss_history_from_dict(loss_hist, method, dataset, seq_len, seed, output_dir)
            if path:
                print(f"  Saved: {path}")
    else:
        print("No loss_histories.jsonl found.")
        # Check for individual experiment result files
        import glob
        result_files = glob.glob(os.path.join(args.results_dir, "*_results.json"))
        result_files += glob.glob(os.path.join(args.results_dir, "*", "*_results.json"))
        if result_files:
            print(f"Found {len(result_files)} result files, but loss history needs adapter access")
        else:
            print("No result files found either.")


def save_loss_history_from_dict(loss_hist, method, dataset, seq_len, seed, output_dir):
    """Save loss history from dict."""
    hist = {k: v for k, v in loss_hist.items() if v}
    if not hist:
        return None

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{method}_seed{seed}_loss_history.json")
    data = {
        "method": method,
        "dataset": dataset,
        "seq_len": seq_len,
        "seed": seed,
        "loss_history": hist,
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # Also save npz
    npz_path = os.path.join(output_dir, f"{method}_seed{seed}_loss.npz")
    try:
        np.savez(npz_path, **{
            k: np.array([[int(s), float(v)] for s, v in vals])
            for k, vals in hist.items()
        })
    except:
        pass

    return json_path


if __name__ == "__main__":
    main()
