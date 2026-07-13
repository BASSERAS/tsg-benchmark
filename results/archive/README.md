# Results Directory

This directory stores all benchmark output files. Most files are gitignored (see `.gitignore`); only documentation placeholders are tracked.

---

## File Inventory

| File | Contents | Tracked? |
|------|----------|----------|
| `all_results.jsonl` | Raw per-seed experiment results in JSON Lines format, one JSON object per (method, dataset, seq_len, seed) combination | No (gitignored) |
| `final_all_metrics.json` | Consolidated metric results for all completed experiments | No |
| `{dataset}_{seq_len}.csv` | Per-method table with mean +/- std across 5 seeds, CSV format | No |
| `{dataset}_{seq_len}.md` | Same as CSV but in markdown format | No |
| `aggregate_ranking.csv` | Ranking of methods by mean Tier-A metric value across all datasets/seq_lens | No |
| `aggregate_ranking.md` | Markdown version of aggregate ranking | No |
| `failures.csv` / `failures.md` | List of failed experiments with error messages | No |
| `progress_table.md` | Human-readable progress table showing which experiments completed | No |
| `{prefix}_results.json` | Intermediate result snapshots during development (e.g. `pt_valid_results.json`, `tf1_sines_results.json`) | No |
| `check*.png` / `check*.txt` | Pre-flight verification plots and outputs from method validation | Yes (check files are tracked) |
| `heston_paths.png` | Sample Heston price paths visualization | Yes |
| `heston_verification.png` | Heston verification plot | Yes |
| `README.md` | This file | Yes |

---

## JSONL Format

Each line in `all_results.jsonl` is a JSON object with these fields:

```json
{
  "method": "csdi",
  "dataset": "sines",
  "seq_len": 24,
  "seed": 0,
  "status": "OK",
  "train_time_sec": 177.7,
  "num_parameters": 609617,
  "path_mmd2": 0.001162,
  "terminal_mmd2": 0.000400,
  "increment_mmd2": 0.000184,
  "volatility_mmd": 0.018683,
  "terminal_swd": 0.023115,
  "path_swd": 0.035265,
  "cov_error": 0.016085,
  "mean_rmse": 0.052344,
  "std_error": 0.005603,
  "kurtosis_error": 0.031940,
  "acf_err_abs": 0.034626,
  "acf_err_sq": 0.038085,
  "discriminative_score": NaN,
  "predictive_score": NaN,
  "teacher_sigma_corr": 0.9156,
  "teacher_sigma_rmse": 1.1257
}
```

Metric keys follow the naming pattern in `metrics.py`:

| Key | Metric ID | Description |
|-----|-----------|-------------|
| `path_mmd2` | A1 | Full joint-path MMD^2 |
| `terminal_mmd2` | A2 | Terminal MMD^2 |
| `increment_mmd2` | A3 | Increment MMD^2 |
| `volatility_mmd` | A4 | Volatility-discrepancy MMD |
| `terminal_swd` | A5 | Terminal Sliced Wasserstein Distance |
| `path_swd` | A6 | Path Sliced Wasserstein Distance |
| `cov_error` | A7 | Terminal Covariance Error |
| `mean_rmse` | A8 | Terminal Mean RMSE |
| `std_error` | A9 | Return Std Error |
| `kurtosis_error` | A10 | Return Kurtosis Error |
| `acf_err_abs` | A11 | ACF Error (absolute returns) |
| `acf_err_sq` | A12 | ACF Error (squared returns) |
| `discriminative_score` | A13 | Discriminative Score |
| `predictive_score` | A14 | Predictive Score (TSTR) |
| `teacher_sigma_corr` | A15 | Teacher-Sigma Correlation (Heston only) |
| `teacher_sigma_rmse` | A15 | Teacher-Sigma RMSE (Heston only) |

---

## How to Load Results

```python
import pandas as pd

# All per-seed results
df = pd.read_json("all_results.jsonl", lines=True)
print(f"{len(df)} experiments loaded")
print(df.groupby("method")["status"].value_counts())

# Aggregate table for a specific dataset and seq_len
table = pd.read_csv("sines_24.csv")
print(table)

# List failures
failures = pd.read_csv("failures.csv")
print(failures)
```

---

## Notes

- Aggregate tables are only produced for full sweeps (not in `--small` mode).
- The `results/` directory is gitignored. Intermediate snapshots (`*_results.json`) were generated during development and may not reflect final sweep results.
- Plots in this directory are documentation artifacts and are tracked.
