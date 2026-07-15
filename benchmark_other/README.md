# Results Directory

This directory stores all benchmark output files. The canonical results file
is `sines_seq24_results.json` for the Sines dataset at sequence length 24.

## Files

| File | Contents | Tracked? |
|------|----------|----------|
| `sines_seq24_results.json` | Merged results for all 8 methods x 5 seeds (40 experiments) on Sines seq_len=24 | Yes |
| `archive/` | Previous result files, moved here for history | Yes |

The `archive/` directory contains intermediate snapshots generated during
development (`pt_valid_results.json`, `tf1_sines_results.json`, etc.) and
verification artifacts (check plots, progress tables). They are preserved
for reference but `sines_seq24_results.json` is the authoritative source.

## File Format

`sines_seq24_results.json` is a JSON array. Each element is an experiment
result with 21 fields:

```json
{
  "method": "csdi",
  "dataset": "sines",
  "seq_len": 24,
  "seed": 0,
  "status": "OK",
  "params": 609617,
  "time": 36.5,
  "path_mmd2": 0.0071549,
  "terminal_mmd2": 0.0074616,
  "increment_mmd2": 0.0004924,
  "volatility_mmd": 0.082365,
  "terminal_swd": 0.093396,
  "path_swd": 0.086648,
  "cov_error": 0.054605,
  "mean_rmse": 0.224796,
  "std_error": 0.019550,
  "kurtosis_error": 0.551808,
  "acf_err_abs": 0.064284,
  "acf_err_sq": 0.066727,
  "discriminative_score": 0.24375,
  "predictive_score": 0.34692
}
```

### Metadata fields

| Field | Description |
|-------|-------------|
| `method` | Method name: `csdi`, `fourierflows`, `gtgan`, `diffusionts`, `tsdiff`, `timegan`, `rgan`, `timevae` |
| `dataset` | Dataset name (`sines` for this file) |
| `seq_len` | Sequence length |
| `seed` | Random seed (0-4) |
| `status` | `"OK"` or `"FAILED"` |
| `params` | Number of trainable parameters |
| `time` | Training time in seconds |

### Metric fields (14 total)

See the repo README for detailed metric descriptions.

| Key | ID | Description | Lower is |
|-----|----|-------------|----------|
| `path_mmd2` | A1 | Full joint-path MMD^2 | Better |
| `terminal_mmd2` | A2 | Terminal (final step) MMD^2 | Better |
| `increment_mmd2` | A3 | Increment/return MMD^2 | Better |
| `volatility_mmd` | A4 | Volatility stylized facts MMD | Better |
| `terminal_swd` | A5 | Terminal Sliced Wasserstein Distance | Better |
| `path_swd` | A6 | Path Sliced Wasserstein Distance | Better |
| `cov_error` | A7 | Terminal covariance Frobenius error | Better |
| `mean_rmse` | A8 | Terminal mean RMSE | Better |
| `std_error` | A9 | Return std absolute error | Better |
| `kurtosis_error` | A10 | Return excess kurtosis error | Better |
| `acf_err_abs` | A11 | ACF error on absolute returns | Better |
| `acf_err_sq` | A12 | ACF error on squared returns | Better |
| `discriminative_score` | A13 | |0.5 - GRU classifier accuracy| | Better |
| `predictive_score` | A14 | Train-on-Synthetic-Test-on-Real MAE | Better |

> **Note on A13/A14:** These require TensorFlow 1.x (from the TimeGAN repo).
> Methods run in the PyTorch environment get NaN for `tsdiff`, or may have
> values from a separate post-processing step using the TF1 environment.
>
> `tsdiff` has known issues (mode collapse) — see `KNOWN_ISSUES.md`.

## How to Load

```python
import json, pandas as pd

# Load all results
with open("sines_seq24_results.json") as f:
    data = json.load(f)

# As a DataFrame
df = pd.DataFrame(data)
print(f"{len(df)} experiments loaded")
print(df.groupby("method")["status"].value_counts())

# Per-method aggregate (mean across seeds)
agg = df.groupby("method")[[
    "path_mmd2", "terminal_mmd2", "increment_mmd2", "volatility_mmd",
    "terminal_swd", "path_swd", "cov_error", "mean_rmse", "std_error",
    "kurtosis_error", "acf_err_abs", "acf_err_sq",
    "discriminative_score", "predictive_score"
]].agg(["mean", "std"])
print(agg.to_string())
```
