# TSG SOTA Benchmark -- Unified Time Series Generation Benchmark

A unified, reproducible benchmarking pipeline that compares **8 state-of-the-art time series generation methods** under identical conditions across **5 datasets** and **3 sequence lengths** with **5 random seeds**, evaluated on **14 Tier-A metrics**.

**Goal:** Provide the first standardized, apples-to-apples comparison of time series generative models -- from GANs and VAEs to normalizing flows and diffusion models -- on a common grid with fixed training budgets and identical preprocessing.

---

## Methods

| # | Method | Family | Venue | Paper | Official GitHub | Parameters (Sines, seq=24) |
|---|--------|--------|-------|-------|-----------------|---------------------------|
| 1 | TimeGAN | GAN | NeurIPS 2019 | [Yoon et al.](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks) | [jsyoon0823/TimeGAN](https://github.com/jsyoon0823/TimeGAN) | 48,606 |
| 2 | RGAN / RCGAN | Recurrent GAN | arXiv 2017 | [Esteban et al.](https://arxiv.org/abs/1706.02633) | [ratschlab/RGAN](https://github.com/ratschlab/RGAN) | 6,198 |
| 3 | GT-GAN | GAN + Neural ODE | NeurIPS 2022 | [Jeon et al.](https://arxiv.org/abs/2210.02040) | [Jinsung-Jeon/GT-GAN](https://github.com/Jinsung-Jeon/GT-GAN) | 12,168 |
| 4 | TimeVAE | VAE | arXiv 2021 | [Desai et al.](https://arxiv.org/abs/2111.08095) | [abudesai/timeVAE](https://github.com/abudesai/timeVAE) | 181,816 |
| 5 | Fourier Flows | Normalizing Flow | ICLR 2021 | [Alaa et al.](https://openreview.net/forum?id=u3skL0URlL) | [ahmedmalaa/Fourier-flows](https://github.com/ahmedmalaa/Fourier-flows) | ~274K |
| 6 | CSDI | Conditional Diffusion | NeurIPS 2021 | [Tashiro et al.](https://arxiv.org/abs/2107.03575) | [ermongroup/CSDI](https://github.com/ermongroup/CSDI) | ~610K |
| 7 | TSDiff | Unconditional Diffusion | NeurIPS 2023 | [Kollovieh et al.](https://arxiv.org/abs/2307.11494) | [amazon-science/unconditional-time-series-diffusion](https://github.com/amazon-science/unconditional-time-series-diffusion) | 192,582 |
| 8 | Diffusion-TS | Interpretable Diffusion | ICLR 2024 | [Yuan & Qiao](https://arxiv.org/abs/2403.01742) | [Y-debug-sys/Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS) | 449,373 |

**Note on TSDiff:** This method currently **FAILS** on this benchmark due to a `context_length=0` incompatibility with GluonTS that causes mode collapse. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for details.

---

## Datasets

| Dataset | Type | Samples | Features | Description |
|---------|------|---------|----------|-------------|
| **Sines** | Synthetic | 10,000 | 5 | Independent sine waves, f ~ U(0, 0.1), phi ~ U(0, 2pi), normalized to [0, 1] |
| **Stocks** | Real (GOOG) | Sliding windows | 5 | GOOG daily OHLCV (Open, High, Low, Close, Adj_Close), TimeGAN protocol, stride 1 |
| **Energy** | Real (UCI) | Sliding windows | 28 | Appliances energy prediction (temperature, humidity, pressure, etc.), TimeGAN protocol, stride 1 |
| **Heston** | Synthetic | 10,000 | 1 + latent variance | Stochastic volatility model with Euler-Maruyama full-truncation scheme (mu=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7) |
| **Sinusoidal mixture** | Synthetic | 5,000 | 1 | K=3 components at SNR=20dB (K in {2,3,5} and SNR in {10dB, 20dB, inf} available parametrically) |

> **Preprocessing:** All datasets undergo per-feature min-max normalization to [0, 1] (fit on training set only). An 80/20 train/test split is applied identically across all methods using the same seed. See [docs/datasets.md](docs/datasets.md) for full details.

---

## Metrics (Tier A)

All metrics compare generated (synthetic) samples against real test data. **Lower is better** except for `teacher_sigma_corr` (higher is better).

| ID | Metric | Key | Type | Description |
|----|--------|-----|------|-------------|
| A1 | Path MMD^2 | `path_mmd2` | **Distribution** | Multi-scale RBF MMD on full joint path distribution (flattened T x d) |
| A2 | Terminal MMD^2 | `terminal_mmd2` | **Distribution** | Multi-scale RBF MMD restricted to final time step |
| A3 | Increment MMD^2 | `increment_mmd2` | **Distribution** | Multi-scale RBF MMD on first differences (returns) |
| A4 | Volatility MMD | `volatility_mmd` | **Stylized** | Weighted sum of MMD^2 over 9 stylized fact views of returns |
| A5 | Terminal SWD | `terminal_swd` | **Distribution** | Sliced 1-Wasserstein distance at final time step (50 projections) |
| A6 | Path SWD | `path_swd` | **Distribution** | Mean of sliced 1-Wasserstein across all time steps |
| A7 | Cov Error | `cov_error` | **Moment** | Frobenius norm of terminal covariance matrix difference |
| A8 | Mean RMSE | `mean_rmse` | **Moment** | L2 norm of terminal mean vector difference |
| A9 | Std Error | `std_error` | **Moment** | Absolute difference of increment standard deviation |
| A10 | Kurtosis Error | `kurtosis_error` | **Moment** | Absolute difference of increment excess kurtosis (Fisher) |
| A11 | ACF Error (abs) | `acf_err_abs` | **Temporal** | Mean absolute ACF error on absolute returns at lags {1, 2, 5, 10}, per-sample per-feature |
| A12 | ACF Error (sq) | `acf_err_sq` | **Temporal** | Mean absolute ACF error on squared returns at lags {1, 2, 5, 10}, per-sample per-feature |
| A13 | Discriminative | `discriminative_score` | **Post-hoc** | |0.5 - GRU classifier test accuracy|; 0 is perfect, requires TF 1.x |
| A14 | Predictive (TSTR) | `predictive_score` | **Post-hoc** | Train on synthetic, evaluate on real (one-step-ahead GRU, MAE); requires TF 1.x |
| A15 | Teacher-Sigma | `teacher_sigma_corr` / `teacher_sigma_rmse` | **Heston only** | Correlation and RMSE between estimated rolling vol and true latent sqrt(v(t)) |

> **Metric categories:**
> - **Distribution:** Measures how well the full distribution matches (MMD, SWD)
> - **Moment:** Measures specific statistical moments (mean, variance, kurtosis, covariance)
> - **Temporal:** Measures time-dependent structure (autocorrelation)
> - **Stylized:** Combines multiple views for financial stylized facts
> - **Post-hoc:** Trains a secondary classifier/predictor to evaluate utility
>
> **Important:** A13/A14 require TensorFlow 1.x (from the TimeGAN repository). When computed from the PyTorch environment, they return NaN. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md#4-a13a14-discriminativepredictive-scores-need-tf1) for the workaround.

---

## Setup Instructions

### Prerequisites

- **Conda** (Miniconda recommended). The repository includes a `miniconda3/` installation at the project root.
- **Git** for cloning method repositories.
- **NVIDIA GPU** with CUDA 11.8+ recommended (Fourier Flows can run on CPU).

### Step-by-Step

```bash
# 1. Clone this repository
git clone https://github.com/BASSERAS/tsg-benchmark.git
cd tsg_benchmark

# 2. Clone method repositories into repos/
bash scripts/clone_repos.sh

# 3. Create conda environments
#    The benchmark uses three separate conda environments to handle
#    conflicting framework requirements:
#
#    - tf1_env:    TimeGAN, RGAN (TensorFlow 1.x via tf.compat.v1)
#    - timevae_env: TimeVAE (TensorFlow 2.16)
#    - common_pt:  GT-GAN, Fourier Flows, CSDI, TSDiff, Diffusion-TS (PyTorch 2.x)
#
bash scripts/setup_envs.sh

# 4. Generate data (data is generated on-the-fly by run_benchmark.py)
#    No separate data download is needed for synthetic datasets.
#    For stocks/energy, ensure repos/TimeGAN/data/ contains stock_data.csv
#    and energy_data.csv (these come with the TimeGAN repository).

# 5. Run a small validation test (5% data, 1 seed, 2 methods, 1 dataset)
./miniconda3/envs/common_pt/bin/python run_benchmark.py --small \
    --methods timegan,csdi --datasets sines --seq-len 24

# 6. Full sweep (all methods, datasets, seq_lens, 5 seeds)
#    WARNING: This will take several hours on 4 GPUs.
./miniconda3/envs/common_pt/bin/python run_benchmark.py

# 7. Compute metrics
#    Metrics are computed automatically during the sweep.
#    For A13/A14 (TF 1.x), post-process from tf1_env:
conda run -n tf1_env python -c "
from metrics import compute_discriminative_score, compute_predictive_score
import numpy as np, json, os
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
# Load generated samples and compute scores for each method...
"
```

### Conda Environment Map

| Method | Conda Env | Framework | Python |
|--------|-----------|-----------|--------|
| TimeGAN | `tf1_env` | TensorFlow 2.2 (TF1 compat) | 3.7 |
| RGAN | `tf1_env` | TensorFlow 2.2 (TF1 compat) | 3.7 |
| GT-GAN | `common_pt` | PyTorch 2.x + CUDA 11.8 | 3.10 |
| TimeVAE | `timevae_env` | TensorFlow 2.16 | 3.12 |
| Fourier Flows | `common_pt` | PyTorch (CPU only) | 3.10 |
| CSDI | `common_pt` | PyTorch 2.x + CUDA 11.8 | 3.10 |
| TSDiff | `common_pt` | PyTorch + GluonTS | 3.10 |
| Diffusion-TS | `common_pt` | PyTorch 2.x + CUDA 11.8 | 3.10 |

---

## Usage

```bash
# Full sweep (all methods, datasets, seq_lens, 5 seeds)
python run_benchmark.py

# Small validation run (5% data, 1 seed)
python run_benchmark.py --small

# Subset of methods and datasets
python run_benchmark.py --methods timegan,csdi --datasets sines,heston

# Single sequence length
python run_benchmark.py --seq-len 24

# Dry run (alias for --small)
python run_benchmark.py --dry-run

# Specify GPUs
python run_benchmark.py --gpus 0,1

# Custom output directory
python run_benchmark.py --output /path/to/results
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dry-run` | `False` | Run with 5% data (smoke test) |
| `--small` | `False` | Alias for --dry-run |
| `--methods` | All 8 | Comma-separated method list |
| `--datasets` | All 5 | Comma-separated dataset list |
| `--seq-len` | All 3 | Single seq_len (24, 64, or 128) |
| `--gpus` | `0,1,2,3` | GPU IDs to use, space-separated |
| `--output` | `results/` | Results directory |

---

## Directory Structure

```
tsg_benchmark/
├── run_benchmark.py              # Main orchestration script
├── metrics.py                    # All Tier-A metric implementations
├── requirements.txt              # Python dependencies for common_pt env
├── KNOWN_ISSUES.md               # Known failures and workarounds
├── verification_report.md        # Code audit findings
├── .gitignore                    # Excludes repos/, miniconda3/, results/, __pycache__/
│
├── data/
│   ├── __init__.py
│   └── datasets.py               # Data generation, loading, preprocessing
│
├── adapters/
│   ├── __init__.py
│   ├── timegan_adapter.py        # Wrapper for TimeGAN
│   ├── rgan_adapter.py           # Wrapper for RGAN/RCGAN
│   ├── gtgan_adapter.py          # Wrapper for GT-GAN (neural ODE)
│   ├── timevae_adapter.py        # Wrapper for TimeVAE
│   ├── fourierflows_adapter.py   # Wrapper for Fourier Flows
│   ├── csdi_adapter.py           # Wrapper for CSDI
│   ├── tsdiff_adapter.py         # Wrapper for TSDiff (FAILED)
│   ├── tsdiff_patch.py           # GluonTS compatibility patch
│   └── diffusionts_adapter.py    # Wrapper for Diffusion-TS
│
├── scripts/
│   ├── clone_repos.sh            # Clone all 8 method repositories
│   ├── setup_envs.sh             # Create conda environments
│   ├── run_sweep.sh              # Convenience script for full sweep
│   └── plot_results.py           # Result visualization
│
├── repos/                        # Cloned method repositories (gitignored)
│   ├── TimeGAN/                  # TimeGAN (jsyoon0823)
│   ├── RGAN/                     # RGAN (ratschlab)
│   ├── GT-GAN/                   # GT-GAN (Jinsung-Jeon)
│   ├── timeVAE/                  # TimeVAE (abudesai)
│   ├── Fourier-flows/            # Fourier Flows (ahmedmalaa)
│   ├── CSDI/                     # CSDI (ermongroup)
│   ├── tsdiff/                   # TSDiff (amazon-science)
│   └── Diffusion-TS/             # Diffusion-TS (Y-debug-sys)
│
├── miniconda3/                   # Conda installation (gitignored)
│
├── docs/
│   ├── metrics.md                # Detailed metric documentation
│   ├── datasets.md               # Dataset generation details
│   └── protocol.md               # Experimental protocol
│
├── results/                      # Benchmark output (gitignored except README)
│   ├── README.md                 # Results file format documentation
│   ├── all_results.jsonl         # Raw per-seed results
│   ├── {dataset}_{seq_len}.csv   # Aggregate tables
│   └── check*.png                # Pre-flight verification plots
│
└── envs/                         # Environment config templates
```

---

## Results

### Current Status

| Method | Sines | Stocks | Energy | Heston | Sinusoidal |
|--------|-------|--------|--------|--------|------------|
| TimeGAN | OK | OK | OK | OK | OK |
| RGAN | OK | OK | OK | OK | OK |
| GT-GAN | OK | OK | OK | OK | OK |
| TimeVAE | OK | OK | OK | OK | OK |
| Fourier Flows | OK | OK | OK | OK | OK |
| CSDI | OK | OK | OK | OK | OK |
| TSDiff | FAILED | FAILED | FAILED | FAILED | FAILED |
| Diffusion-TS | OK | OK | OK | OK | OK |

### Validation Results (Sines, seq_len=24)

The benchmark was validated on the Sines dataset (seq_len=24) with all 7 non-failed methods across 5 seeds. Full results including all 15 Tier-A metrics are in `results/full_sines_results.json` and `results/tf1_sines_results.json`.

### How to Read Results

```python
import pandas as pd

# Load all per-seed results
df = pd.read_json("results/all_results.jsonl", lines=True)

# Aggregate table for a specific dataset/seq_len
table = pd.read_csv("results/sines_24.csv")
print(table)

# List failures
failures = pd.read_csv("results/failures.csv")
print(failures)
```

For detailed format specifications, see [results/README.md](results/README.md).

---

## Key Findings

1. **TSDiff is non-functional** for unconditional generation -- the GluonTS forecasting framework `context_length=0` workaround causes mode collapse across all datasets and seeds.

2. **GT-GAN struggles with kurtosis** -- the Neural CDE + CNF architecture produces high kurtosis error (~44 on Sines vs <2 for all others), likely due to limited ODE solver steps.

3. **Environment complexity matters** -- three separate conda environments are needed to handle TF 1.x (TimeGAN, RGAN), TF 2.x (TimeVAE), and PyTorch (remaining methods) framework conflicts.

4. **A13/A14 require TF 1.x** -- these post-hoc metrics from TimeGAN's evaluation suite cannot run in the PyTorch environment and must be post-processed in the tf1_env.

For complete details on known issues and workarounds, see [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

---

## Known Issues Summary

| Issue | Severity | Status |
|-------|----------|--------|
| TSDiff mode collapse (GluonTS context_length=0) | **FAILED** | No workaround |
| GT-GAN high kurtosis error | **Known limitation** | Accept as-is |
| Double normalization on stock/energy | **Major bug** | Documented |
| TimeGAN sample() returns cached data | **Major bug** | Documented |
| ACF metric bug (per-sample fix applied) | **Major** | Fixed in code |
| A13/A14 NaN in PyTorch env | **Medium** | Use tf1_env workaround |
| Missing batch_size guards (3 adapters) | **Medium** | Affects --small mode |

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) and [verification_report.md](verification_report.md) for the full audit.

---

## References

- [Metrics documentation](docs/metrics.md) -- detailed explanation of all Tier-A metrics
- [Datasets documentation](docs/datasets.md) -- dataset generation and preprocessing
- [Protocol documentation](docs/protocol.md) -- experimental protocol
- [Known Issues](KNOWN_ISSUES.md) -- method failures and workarounds
- [Verification Report](verification_report.md) -- code audit with major, medium, and low severity findings
- [Results Format](results/README.md) -- results file format documentation

### Paper References

1. Yoon, J., Jarrett, D., & van der Schaar, M. (2019). Time-series Generative Adversarial Networks. *NeurIPS 2019*.
2. Esteban, C., Hyland, S.L., & Ratsch, G. (2017). Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs. *arXiv:1706.02633*.
3. Jeon, J., Kim, J., Kim, H., Kang, U., & Kang, J. (2022). GT-GAN: Generative Adversarial Networks for Time Series with Neural ODE. *NeurIPS 2022*.
4. Desai, A., Freeman, C., Wang, Z., & Beaver, I. (2021). TimeVAE: A Variational Auto-Encoder for Multivariate Time Series Generation. *arXiv:2111.08095*.
5. Alaa, A., Chan, A., & van der Schaar, M. (2021). Generative Time Series Modeling with Fourier Flows. *ICLR 2021*.
6. Tashiro, Y., Song, J., & Ermon, S. (2021). CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation. *NeurIPS 2021*.
7. Kollovieh, P., et al. (2023). Unconditional Time Series Diffusion Models using S4. *NeurIPS 2023 (TTS Workshop)*.
8. Yuan, C. & Qiao, H. (2024). Diffusion-TS: Interpretable Diffusion for General Time Series Generation. *ICLR 2024*.

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{tsg-sota-benchmark,
  author = {Basseras, Theo and others},
  title = {TSG SOTA Benchmark: A Unified Benchmark for Time Series Generation},
  year = {2026},
  howpublished = {\url{https://github.com/BASSERAS/tsg-benchmark}}
}
```

---

## License

This project is for academic research use. Each method repository has its own license (see respective repos/ directories). The benchmark code, adapters, metrics, and documentation are provided under the MIT License.
