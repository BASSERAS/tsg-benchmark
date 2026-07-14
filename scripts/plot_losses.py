#!/usr/bin/env python3
"""
Generate loss evolution plots for all completed experiments.

Reads from results/*_loss_history.json and produces PNG plots.
"""

import json
import os
import sys
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def plot_loss_history(loss_data, method, dataset, seq_len, seed, output_dir):
    """Plot loss curves for one experiment."""
    if not loss_data or not any(v for v in loss_data.values() if v):
        return

    title = f"{method.upper()} — {dataset} seq_len={seq_len} seed={seed}"

    n_plots = sum(1 for v in loss_data.values() if v)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0
    for loss_name, values in loss_data.items():
        if not values:
            continue
        values = sorted(values, key=lambda x: x[0])
        steps = [v[0] for v in values]
        vals = [v[1] for v in values]

        ax = axes[ax_idx]
        ax.plot(steps, vals, "b-", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel(loss_name)
        ax.set_title(loss_name)
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fname = f"{method}_s{seed}_loss.png"
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_plots():
    """Find all loss history files and generate plots."""
    loss_files = glob(os.path.join(RESULTS_DIR, "*_loss_history.json"))

    if not loss_files:
        print("No loss history files found in results/")
        print("Expected pattern: results/{dataset}_{seq_len}/{method}_seed{N}_loss_history.json")
        print("Searching recursively...")
        loss_files = glob(os.path.join(RESULTS_DIR, "**", "*loss_history*"), recursive=True)
        if not loss_files:
            print("Still no loss files found.")
            # Check adapter loss tracking stored inline
            print("Loss tracking is stored in-memory within adapter objects.")
            print("To save, run: python save_loss_histories.py")
            return

    print(f"Found {len(loss_files)} loss history files")

    for lf in sorted(loss_files):
        with open(lf) as f:
            data = json.load(f)

        method = data.get("method", "unknown")
        dataset = data.get("dataset", "unknown")
        seq_len = data.get("seq_len", 0)
        seed = data.get("seed", 0)
        loss_hist = data.get("loss_history", {})

        parts = lf.replace(RESULTS_DIR, "").strip("/").split("/")
        output_dir = os.path.join(RESULTS_DIR, "loss_plots")
        if len(parts) > 1:
            output_dir = os.path.join(RESULTS_DIR, parts[0], "loss_plots")
        else:
            output_dir = os.path.join(RESULTS_DIR, f"{dataset}_seq{seq_len}_loss_plots")

        path = plot_loss_history(loss_hist, method, dataset, seq_len, seed, output_dir)
        if path:
            print(f"  Saved: {path}")

    print(f"\nDone. Plots saved under {RESULTS_DIR}/loss_plots/")


def plot_from_adapter_output():
    """Generate plots from adapter loss output saved to JSONL."""
    loss_jsonl = os.path.join(RESULTS_DIR, "loss_histories.jsonl")
    if not os.path.exists(loss_jsonl):
        return

    from collections import defaultdict
    experiments = defaultdict(list)

    with open(loss_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                exp = json.loads(line)
                key = (exp["method"], exp["dataset"], exp["seq_len"], exp["seed"])
                experiments[key].append(exp)

    for (method, dataset, seq_len, seed), entries in experiments.items():
        # Aggregate loss entries
        loss_data = {}
        for entry in entries:
            for loss_name, values in entry.get("loss_history", {}).items():
                if loss_name not in loss_data:
                    loss_data[loss_name] = []
                loss_data[loss_name].extend(values)

        output_dir = os.path.join(RESULTS_DIR, f"{dataset}_seq{seq_len}_loss_plots")
        path = plot_loss_history(loss_data, method, dataset, seq_len, seed, output_dir)
        if path:
            print(f"  Plot: {path}")


def main():
    print("Generating loss evolution plots...")
    generate_all_plots()
    plot_from_adapter_output()
    print("Done!")


if __name__ == "__main__":
    main()
