# Reproducing the TSG SOTA Benchmark

Reproduce the unified time series generation benchmark comparing 8 methods
on 14 Tier-A metrics (Sines dataset, sequence length 24, 5 random seeds).

## Prerequisites

- Linux with 4x A100 GPUs (or CPU-only for small runs; GPU strongly recommended)
- Miniconda installed and on PATH
- git
- curl (for conda packages)
- Python 3.7+ support via conda

## Step 1: Clone this repo

```bash
git clone https://github.com/BASSERAS/tsg-benchmark.git
cd tsg-benchmark
```

## Step 2: Clone method repositories

```bash
mkdir -p repos && cd repos
git clone https://github.com/jsyoon0823/TimeGAN.git
git clone https://github.com/ratschlab/RGAN.git
git clone https://github.com/Jinsung-Jeon/GT-GAN.git
git clone https://github.com/abudesai/timeVAE.git
git clone https://github.com/ahmedmalaa/Fourier-flows.git
git clone https://github.com/ermongroup/CSDI.git
git clone https://github.com/amazon-science/unconditional-time-series-diffusion.git
git clone https://github.com/Y-debug-sys/Diffusion-TS.git
cd ..
```

Or use the provided script:

```bash
bash scripts/clone_repos.sh
```

## Step 3: Create conda environments

Three conda environments are required due to incompatible framework versions.

### TF1 environment (TimeGAN, RGAN, and A13/A14 metric computation)

```bash
conda create -y -n tf1_env python=3.7 pip
conda run -n tf1_env pip install tensorflow==2.2.0 numpy scikit-learn pandas tqdm matplotlib scipy
```

### PyTorch environment (CSDI, Fourier-flows, GT-GAN, Diffusion-TS, TSDiff)

```bash
conda create -y -n common_pt python=3.10 pip
conda run -n common_pt pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda run -n common_pt pip install numpy pandas scikit-learn scipy matplotlib seaborn tqdm pyyaml einops opt_einsum
conda run -n common_pt pip install pytorch-lightning==1.9.5 gluonts torchdiffeq controldiffeq linear_attention_transformer pot
```

### timeVAE environment (TensorFlow 2)

```bash
conda create -y -n timevae_env python=3.12 pip
conda run -n timevae_env pip install tensorflow==2.16.1 numpy pandas scikit-learn matplotlib pyyaml
```

> If you installed Miniconda inside the repo (as `./miniconda3/`), replace
> `conda` with `./miniconda3/bin/conda` in all commands above.

## Step 4: Generate synthetic datasets

```bash
conda run -n common_pt python -c "from data.datasets import prepare_dataset; prepare_dataset('sines', 24, 0)"
```

This creates `data/sines_seq24.h5` used by all methods.

## Step 5: Run the benchmark

Run one batch per environment. Each command runs 5 seeds sequentially (0-4)
and outputs per-seed JSON results to `results/`.

### PyTorch methods (CSDI, Fourier-flows, GT-GAN, Diffusion-TS, TSDiff)

```bash
conda run -n common_pt python run_benchmark.py \
    --datasets sines --seq-len 24 \
    --methods csdi,fourierflows,gtgan,diffusionts,tsdiff
```

### TF1 methods (TimeGAN, RGAN)

```bash
conda run -n tf1_env python run_benchmark.py \
    --datasets sines --seq-len 24 \
    --methods timegan,rgan
```

### timeVAE

```bash
conda run -n timevae_env python run_benchmark.py \
    --datasets sines --seq-len 24 \
    --methods timevae
```

> If your conda is inside the repo, use the explicit python paths:
> `./miniconda3/envs/common_pt/bin/python run_benchmark.py ...`

### Expected runtime

- 4x A100 GPUs, all methods in parallel (2 GPUs at a time): ~1 hour
- CPU only: 4-8 hours
- The run_benchmark.py script handles one method at a time by default.
  Parallelize across GPUs using `CUDA_VISIBLE_DEVICES` with independent calls.

## Step 6: Compute all metrics

Metrics A1-A12 are computed automatically during `run_benchmark.py`.

A13 (Discriminative Score) and A14 (Predictive Score / TSTR) require
TensorFlow 1.x and are computed only when running from the `tf1_env`
environment. When running from `common_pt` (PyTorch env), they return NaN.

For PyTorch-run methods (CSDI, Fourier-flows, GT-GAN, Diffusion-TS, TSDiff)
that get NaN for A13/A14, re-run metrics from the TF1 environment:

```bash
conda run -n tf1_env python -c "
import json, sys; sys.path.insert(0, '.')
from metrics import compute_all_metrics
import numpy as np

result_path = 'results/sines_seq24_results.json'
with open(result_path) as f:
    data = json.load(f)

for entry in data:
    if entry['status'] != 'OK': continue
    samples = np.load(f'/tmp/samples_{entry[\"method\"]}_s24_s{entry[\"seed\"]}.npy')
    real = np.load('data/sines_seq24.npy')[:len(samples)]
    # Samples are shape (n_gen, T, d), real is (n_real, T, d)
    if entry['method'] in ('timegan', 'rgan', 'timevae'):
        continue  # already have A13/A14
    scores = compute_all_metrics(samples, real, compute_discriminative=True, compute_predictive=True)
    print(f\"{entry['method']} s{entry['seed']}: DS={scores['discriminative_score']:.4f}, PS={scores['predictive_score']:.4f}\")
"
```

## Step 7: View results

The merged results file is at `results/sines_seq24_results.json`. Each entry
contains: method, dataset, seq_len, seed, status, params, time, and all 14
metric keys (path_mmd2 through predictive_score).

```python
import json, pandas as pd

with open("results/sines_seq24_results.json") as f:
    data = json.load(f)
df = pd.DataFrame(data)
print(df.groupby("method")[["path_mmd2", "terminal_mmd2"]].mean())
```

## Methods and repos (summary)

| # | Method | Family | Repository | Parameters |
|---|--------|--------|------------|------------|
| 1 | TimeGAN | GAN (TF1) | https://github.com/jsyoon0823/TimeGAN | ~49K |
| 2 | RGAN | Recurrent GAN (TF1) | https://github.com/ratschlab/RGAN | ~6K |
| 3 | GT-GAN | GAN + Neural ODE (PT) | https://github.com/Jinsung-Jeon/GT-GAN | ~12K |
| 4 | TimeVAE | VAE (TF2) | https://github.com/abudesai/timeVAE | ~182K |
| 5 | Fourier-flows | Normalizing Flow (PT) | https://github.com/ahmedmalaa/Fourier-flows | ~274K |
| 6 | CSDI | Conditional Diffusion (PT) | https://github.com/ermongroup/CSDI | ~610K |
| 7 | TSDiff | Unconditional Diffusion (PT) | https://github.com/amazon-science/unconditional-time-series-diffusion | ~193K |
| 8 | Diffusion-TS | Interpretable Diffusion (PT) | https://github.com/Y-debug-sys/Diffusion-TS | ~449K |

## Metrics (all 14)

| ID | Key | Description |
|----|-----|-------------|
| A1 | path_mmd2 | Full joint-path MMD^2 |
| A2 | terminal_mmd2 | Terminal (final step) MMD^2 |
| A3 | increment_mmd2 | Increment/returns MMD^2 |
| A4 | volatility_mmd | Volatility stylized facts MMD |
| A5 | terminal_swd | Terminal Sliced Wasserstein Distance |
| A6 | path_swd | Path Sliced Wasserstein Distance |
| A7 | cov_error | Terminal covariance Frobenius error |
| A8 | mean_rmse | Terminal mean RMSE |
| A9 | std_error | Return std absolute error |
| A10 | kurtosis_error | Return excess kurtosis error |
| A11 | acf_err_abs | ACF error on absolute returns |
| A12 | acf_err_sq | ACF error on squared returns |
| A13 | discriminative_score | |0.5 - GRU classifier accuracy| |
| A14 | predictive_score | Train-on-Synthetic-Test-on-Real MAE |

Lower is better for all metrics.

## Known issues

- **TSDiff** currently fails (mode collapse) due to a `context_length=0`
  incompatibility with GluonTS. See `KNOWN_ISSUES.md`.
- **A13/A14** (discriminative/predictive scores) require TensorFlow 1.x.
  Methods run in the PyTorch environment return NaN for these metrics.
  The workaround is to re-run from `tf1_env` with saved sample arrays.
