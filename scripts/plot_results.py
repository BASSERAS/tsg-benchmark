#!/usr/bin/env python3
"""Plot benchmark results from CSV files."""
import os, sys, glob, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--output-dir', default='results/figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read all CSV files
    csv_files = sorted(glob.glob(os.path.join(args.results_dir, '*_*.csv')))
    # Exclude aggregate
    csv_files = [f for f in csv_files if 'aggregate' not in f and 'failure' not in f]

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        print(f"Reading {csv_file}...")
        # Extract dataset and seq_len from filename
        basename = os.path.basename(csv_file).replace('.csv', '')
        print(f"  {len(df)} methods x {len(df.columns)} metrics")

    # Plot: method comparison bar chart for key metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics_to_plot = ['Path MMD²', 'Discriminative Score', 'Predictive Score', 'Std Error']

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            if metric not in df.columns:
                continue
            basename = os.path.basename(csv_file).replace('.csv', '')
            methods = df['Method'].values
            values = []
            for v in df[metric].values:
                if isinstance(v, str) and '±' in v:
                    values.append(float(v.split('±')[0].strip()))
                elif isinstance(v, (int, float)):
                    values.append(v)
                else:
                    values.append(0)
            bars = ax.bar([f"{m[:4]}" for m in methods], values, alpha=0.7, label=basename)
        ax.set_title(f'{metric} by Method')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([m[:6] for m in df['Method']], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'metrics_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved {os.path.join(args.output_dir, 'metrics_comparison.png')}")

if __name__ == '__main__':
    main()
