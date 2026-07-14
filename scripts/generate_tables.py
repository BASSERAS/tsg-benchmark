#!/usr/bin/env python3
"""
Generate summary tables for completed benchmark experiments.

Usage:
    python scripts/generate_tables.py [--results-dir results]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# All 14 Tier-A metrics
METRICS = [
    "path_mmd2", "terminal_mmd2", "increment_mmd2", "volatility_mmd",
    "terminal_swd", "path_swd", "cov_error", "mean_rmse",
    "std_error", "kurtosis_error", "acf_err_abs", "acf_err_sq",
    "discriminative_score", "predictive_score",
]

METRIC_LABELS = {
    "path_mmd2": "Path MMD²",
    "terminal_mmd2": "Terminal MMD²",
    "increment_mmd2": "Increment MMD²",
    "volatility_mmd": "Volatility MMD",
    "terminal_swd": "Terminal SWD",
    "path_swd": "Path SWD",
    "cov_error": "Cov Error",
    "mean_rmse": "Mean RMSE",
    "std_error": "Std Error",
    "kurtosis_error": "Kurtosis Error",
    "acf_err_abs": "ACF Err (abs)",
    "acf_err_sq": "ACF Err (sq)",
    "discriminative_score": "Discriminative",
    "predictive_score": "Predictive",
    "teacher_sigma_corr": "Teacher-σ Corr",
    "teacher_sigma_rmse": "Teacher-σ RMSE",
}


def load_results(results_dir):
    """Load all experiment results from JSONL and JSON files."""
    all_results = []

    # Load from JSONL
    jsonl_path = os.path.join(results_dir, "all_results.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_results.append(json.loads(line))
                    except:
                        pass

    # Also check for per-dataset result files
    import glob
    for json_path in glob.glob(os.path.join(results_dir, "*_results.json")):
        with open(json_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                for entry in data:
                    if entry not in all_results:
                        all_results.append(entry)
            elif isinstance(data, dict):
                if data not in all_results:
                    all_results.append(data)

    return all_results


def generate_summary_table(results, output_dir):
    """Generate per-dataset summary tables."""
    if not results:
        print("No results to summarize.")
        return

    # Group by (dataset, seq_len)
    groups = defaultdict(list)
    for r in results:
        key = (r.get("dataset"), r.get("seq_len", 0))
        groups[key].append(r)

    for (dataset, seq_len), group in groups.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset}  seq_len={seq_len}")
        print(f"  {len(group)} experiments ({len([g for g in group if g.get('status')=='OK'])} OK)")
        print(f"{'='*80}")

        # Compute per-method aggregates
        method_groups = defaultdict(list)
        for r in group:
            if r.get("status") == "OK":
                method_groups[r["method"]].append(r)

        if not method_groups:
            print("  No successful experiments.")
            continue

        # Build markdown table
        methods_ordered = ["timegan", "rgan", "gtgan", "timevae", "fourierflows", "csdi", "tsdiff", "diffusionts"]
        methods_present = [m for m in methods_ordered if m in method_groups]

        # Determine which metrics are present
        available_metrics = []
        for m in METRICS:
            if any(m in method_groups[mg][0] for mg in methods_present if method_groups[mg]):
                available_metrics.append(m)
        # Check for Heston-specific metrics
        heston_metrics = ["teacher_sigma_corr", "teacher_sigma_rmse"]
        for hm in heston_metrics:
            if any(hm in method_groups[mg][0] for mg in methods_present if method_groups[mg]):
                available_metrics.append(hm)

        # Generate markdown table
        header = f"### {dataset} seq_len={seq_len}\n\n"
        header += "| Method | " + " | ".join(METRIC_LABELS.get(m, m) for m in available_metrics) + " | Params | Time(s) |\n"
        header += "|--------|" + "|".join("-" * len(METRIC_LABELS.get(m, m)) for m in available_metrics) + "|--------|--------|\n"

        rows = []
        for method in methods_present:
            entries = method_groups[method]
            if not entries:
                continue
            vals = {}
            for m in available_metrics:
                metric_vals = [e.get(m) for e in entries if e.get(m) is not None and not (isinstance(e.get(m), float) and np.isnan(e.get(m)))]
                if metric_vals:
                    mean_v = np.mean(metric_vals)
                    std_v = np.std(metric_vals)
                    vals[m] = (mean_v, std_v)
                else:
                    vals[m] = None

            params = entries[0].get("num_parameters", entries[0].get("params", "N/A"))
            if isinstance(params, float):
                params = f"{int(params):,}"
            elif isinstance(params, int):
                params = f"{params:,}"

            time_mean = np.mean([e.get("train_time_sec", 0) for e in entries])

            row = f"| {method} |"
            for m in available_metrics:
                v = vals[m]
                if v is None:
                    row += " NaN |"
                else:
                    mean_v, std_v = v
                    row += f" {mean_v:.4f}±{std_v:.4f} |"
            row += f" {params} | {time_mean:.1f} |"
            rows.append(row)

            # Print to console
            print(f"  {method}: ", end="")
            for m in available_metrics[:6]:
                v = vals[m]
                if v:
                    print(f"{m}={v[0]:.4f}±{v[1]:.4f} ", end="")
                else:
                    print(f"{m}=NaN ", end="")
            print()

        md_content = header + "\n".join(rows)

        # Save markdown
        ds_tag = f"{dataset}_seq{seq_len}"
        md_path = os.path.join(output_dir, f"{ds_tag}.md")
        with open(md_path, "w") as f:
            f.write(md_content)
        print(f"  Saved: {md_path}")

        # Save CSV
        csv_path = os.path.join(output_dir, f"{ds_tag}.csv")
        csv_lines = ["method," + ",".join(available_metrics) + ",params,time_sec"]
        for method in methods_present:
            entries = method_groups[method]
            if not entries:
                continue
            vals = []
            for m in available_metrics:
                metric_vals = [e.get(m) for e in entries if e.get(m) is not None and not (isinstance(e.get(m), float) and np.isnan(e.get(m)))]
                vals.append(str(np.mean(metric_vals)) if metric_vals else "NaN")
            params = entries[0].get("num_parameters", entries[0].get("params", "N/A"))
            time_m = np.mean([e.get("train_time_sec", 0) for e in entries])
            csv_lines.append(f"{method},{','.join(vals)},{params},{time_m:.1f}")
        with open(csv_path, "w") as f:
            f.write("\n".join(csv_lines))
        print(f"  Saved: {csv_path}")

    # Generate aggregate ranking
    generate_ranking(results, output_dir)


def generate_ranking(results, output_dir):
    """Generate aggregate ranking across all datasets."""
    # Group by (method, dataset, seq_len)
    groups = defaultdict(list)
    for r in results:
        if r.get("status") != "OK":
            continue
        key = (r["method"], r.get("dataset"), r.get("seq_len", 0))
        groups[key].append(r)

    # Compute average rank per method
    method_scores = defaultdict(list)

    for (method, dataset, seq_len), entries in groups.items():
        # Compare against other methods on this dataset/seq_len
        all_methods_here = defaultdict(list)
        for (m2, ds2, sl2), e2 in groups.items():
            if ds2 == dataset and sl2 == seq_len:
                for e in e2:
                    for metric in METRICS:
                        if metric in e and e[metric] is not None and not (isinstance(e[metric], float) and np.isnan(e[metric])):
                            all_methods_here[m2].append((metric, e[metric]))

        # Compute rank for each metric
        for metric in METRICS:
            metric_scores = [(m, np.mean([v for met, v in entries if met == metric]))
                           for m, entries in all_methods_here.items()
                           if any(met == metric for met, _ in entries)]
            metric_scores = [(m, s) for m, s in metric_scores if s is not None and not (isinstance(s, float) and np.isnan(s))]
            if len(metric_scores) < 2:
                continue
            metric_scores.sort(key=lambda x: x[1])  # Lower is better
            for rank, (m, _) in enumerate(metric_scores):
                method_scores[m].append(rank + 1)

    if not method_scores:
        return

    print(f"\n{'='*80}")
    print("AGGREGATE RANKING (average rank across all metrics × datasets)")
    print(f"{'='*80}")

    rankings = [(m, np.mean(scores), np.std(scores), len(scores))
                for m, scores in method_scores.items()]
    rankings.sort(key=lambda x: x[1])

    for rank, (method, avg_rank, std_rank, n) in enumerate(rankings, 1):
        print(f"  {rank}. {method:15s}  avg rank: {avg_rank:.2f} ± {std_rank:.2f}  (based on {n} metric comparisons)")
        # Save to file
        ranking_path = os.path.join(output_dir, "aggregate_ranking.md")
        with open(ranking_path, "w") as f:
            f.write("# Aggregate Ranking\n\n")
            f.write("| Rank | Method | Avg Rank | Std | Comparisons |\n")
            f.write("|------|--------|----------|-----|-------------|\n")
            for rank, (method, avg_rank, std_rank, n) in enumerate(rankings, 1):
                f.write(f"| {rank} | {method} | {avg_rank:.2f} | {std_rank:.2f} | {n} |\n")

    print(f"  Saved: {ranking_path}")

    # Convergence report
    generate_convergence_report(results, output_dir)


def generate_convergence_report(results, output_dir):
    """Generate convergence report."""
    print(f"\n{'='*80}")
    print("CONVERGENCE REPORT")
    print(f"{'='*80}")

    methods = set(r["method"] for r in results if r.get("status") == "OK")
    for method in sorted(methods):
        entries = [r for r in results if r["method"] == method and r.get("status") == "OK"]
        print(f"  {method}: {len(entries)} OK experiments")

    report_path = os.path.join(output_dir, "convergence_report.md")
    with open(report_path, "w") as f:
        f.write("# Convergence Report\n\n")
        f.write("| Method | Experiments | OK | Failed | Convergence |\n")
        f.write("|--------|-------------|----|--------|-------------|\n")
        for method in sorted(methods):
            entries = [r for r in results if r["method"] == method]
            ok_entries = [r for r in entries if r.get("status") == "OK"]
            failed = [r for r in entries if r.get("status") != "OK"]
            f.write(f"| {method} | {len(entries)} | {len(ok_entries)} | {len(failed)} | Tracking in loss plots |\n")

    print(f"  Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--input", default=None, help="Specific JSONL file to load")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if args.input and os.path.exists(args.input):
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except:
                        pass

    print(f"Loaded {len(results)} total experiment results")
    print(f"  OK: {len([r for r in results if r.get('status') == 'OK'])}")
    print(f"  FAILED: {len([r for r in results if r.get('status') != 'OK'])}")

    if not results:
        print("No results loaded.")
        return

    generate_summary_table(results, args.results_dir)


if __name__ == "__main__":
    main()
