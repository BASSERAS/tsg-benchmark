# TSG SOTA Benchmark — Unified Time Series Generation Benchmark

A unified, reproducible benchmarking pipeline that compares 8 state-of-the-art time series generation methods under identical conditions across 5 datasets and 3 sequence lengths.

## Methods

| # | Method | Family | Paper | Official GitHub |
|---|--------|--------|-------|-----------------|
| 1 | TimeGAN | GAN | [Yoon et al., NeurIPS 2019](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks) | [jsyoon0823/TimeGAN](https://github.com/jsyoon0823/TimeGAN) |
| 2 | RGAN / RCGAN | Recurrent GAN | [Esteban et al., arXiv:1706.02633](https://arxiv.org/abs/1706.02633) | [ratschlab/RGAN](https://github.com/ratschlab/RGAN) |
| 3 | GT-GAN | GAN + Neural ODE | [Jeon et al., NeurIPS 2022](https://arxiv.org/abs/2210.02040) | [Jinsung-Jeon/GT-GAN](https://github.com/Jinsung-Jeon/GT-GAN) |
| 4 | TimeVAE | VAE | [Desai et al., arXiv:2111.08095](https://arxiv.org/abs/2111.08095) | [abudesai/timeVAE](https://github.com/abudesai/timeVAE) |
| 5 | Fourier Flows | Normalizing Flow | [Alaa et al., ICLR 2021](https://openreview.net/forum?id=u3skL0URlL) | [ahmedmalaa/Fourier-flows](https://github.com/ahmedmalaa/Fourier-flows) |
| 6 | CSDI | Conditional Diffusion | [Tashiro et al., NeurIPS 2021](https://arxiv.org/abs/2107.03575) | [ermongroup/CSDI](https://github.com/ermongroup/CSDI) |
| 7 | TSDiff | Unconditional Diffusion | [Kollovieh et al., NeurIPS 2023](https://arxiv.org/abs/2307.11494) | [amazon-science/unconditional-time-series-diffusion](https://github.com/amazon-science/unconditional-time-series-diffusion) |
| 8 | Diffusion-TS | Interpretable Diffusion | [Yuan & Qiao, ICLR 2024](https://arxiv.org/abs/2403.01742) | [Y-debug-sys/Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS) |

## Datasets

| Dataset | Type | Samples | Features | Description |
|---------|------|---------|----------|-------------|
| **Sines** | Synthetic | 10,000 | 5 | Independent sine waves, f ~ U(0,0.1), phi ~ U(0,2pi) |
| **Stocks** | Real (GOOG) | Sliding windows | 5 | GOOG daily OHLCV, TimeGAN protocol, stride 1 |
| **Energy** | Real (UCI) | Sliding windows | 28 | Appliances energy prediction, TimeGAN protocol, stride 1 |
| **Heston** | Synthetic | 10,000 | 1 + latent | Stochastic volatility, Euler-Maruyama full-truncation |
| **Sinusoidal mixture** | Synthetic | 5,000 per combo | 1 | K=2,3,5 components x SNR=10dB,20dB,inf |

## Metrics (Tier A)

| ID | Metric | Description |
|----|--------|-------------|
| A1 | Path MMD^2 | Multi-scale RBF kernel on full joint path distribution |
| A2 | Terminal MMD^2 | Multi-scale RBF kernel restricted to last timestep |
| A3 | Increment MMD^2 | Multi-scale RBF kernel on first differences |
| A4 | Volatility MMD | Weighted sum over 9 stylized fact views |
| A5 | Terminal SWD | Sliced 1-Wasserstein at t_N, 50 projections |
| A6 | Path SWD | Mean of Terminal SWD across all timesteps |
| A7 | Cov Error | Frobenius norm of terminal covariance difference |
| A8 | Mean RMSE | L2 norm of terminal mean difference |
| A9 | Std Error | Absolute difference of increment standard deviation |
| A10 | Kurtosis Error | Absolute difference of increment excess kurtosis |
| A11 | ACF Error (abs) | Mean absolute ACF error on abs returns |
| A12 | ACF Error (sq) | Mean absolute ACF error on squared returns |
| A13 | Discriminative Score | Post-hoc GRU classifier accuracy deviation from 0.5 |
| A14 | Predictive Score (TSTR) | Train on synthetic, evaluate on real, MAE |
| A15 | Teacher-Sigma | Correlation + RMSE of estimated vs true vol (Heston only) |

**Tier B** (reserved for future model-specific metrics): AC Loss, R Loss, Trajectory RMSE, Control RMSE, Cost Gap, Control Correlation, Route Fraction, Control Energy. Not computed for baseline methods.

## Experimental Protocol

- **Grid**: 8 methods x 5 datasets x 3 sequence lengths {24, 64, 128} x 5 seeds {0, 1, 2, 3, 4}
- **Training budget**: 5000 steps at seq_len=24, scaled to length (seq_len=64: 10000, seq_len=128: 15000)
- **Batch size**: 128
- **Hardware**: 4x NVIDIA A100-80GB, parallelized across GPUs
- **Preprocessing**: Per-feature min-max normalization to [0,1], 80/20 train/test split (identical across methods)
- **Output**: Per-(dataset, seq_len) tables with mean +/- std over 5 seeds

## Setup

```bash
# 1. Clone this repository
git clone <repo-url> tsg_benchmark
cd tsg_benchmark

# 2. Clone method repositories into repos/
#    (see repos/ directory for the expected structure)

# 3. Create conda environments
#    The benchmark uses three conda environments:
#    - tf1_env:    TimeGAN, RGAN (TensorFlow 1.x)
#    - timevae_env: TimeVAE (TensorFlow 2.x)
#    - common_pt:  GT-GAN, Fourier Flows, CSDI, TSDiff, Diffusion-TS (PyTorch)
#
#    A miniconda3 installation is included at the project root.
#    Run scripts/setup_envs.sh to create all environments.

# 4. Run the benchmark (see Usage below)
```

### Conda Environment Map

| Method | Conda Env | Framework |
|--------|-----------|-----------|
| TimeGAN | `tf1_env` | TensorFlow 1.15 |
| RGAN | `tf1_env` | TensorFlow 1.x (compat.v1) |
| GT-GAN | `common_pt` | PyTorch |
| TimeVAE | `timevae_env` | TensorFlow 2.x |
| Fourier Flows | `common_pt` | PyTorch (CPU only) |
| CSDI | `common_pt` | PyTorch |
| TSDiff | `common_pt` | PyTorch + GluonTS |
| Diffusion-TS | `common_pt` | PyTorch |

## Usage

```bash
# Validation run (Sines, seq_len=24, 1 seed, 5% data)
python run_benchmark.py --small --methods timegan,rgan --datasets sines

# Full sweep (all methods, datasets, seq_lens, 5 seeds)
python run_benchmark.py

# Subset of methods and datasets
python run_benchmark.py --methods timegan,csdi --datasets sines,heston

# Single sequence length
python run_benchmark.py --seq-len 24

# Dry run (alias for --small)
python run_benchmark.py --dry-run
```

## Output Format

Results are saved to the `results/` directory. For each (dataset, seq_len) pair:
- `{dataset}_{seq_len}.csv` — per-method table with mean +/- std across seeds
- `{dataset}_{seq_len}.md` — markdown version of the table

Aggregate files:
- `aggregate_ranking.csv` / `aggregate_ranking.md` — ranking by mean Tier-A metric value
- `failures.csv` / `failures.md` — experiments that failed
- `all_results.jsonl` — raw per-seed results in JSON Lines format

## Project Layout

```
tsg_benchmark/
├── run_benchmark.py          # Main orchestration script
├── metrics.py                # All Tier-A metric implementations
├── data/
│   ├── __init__.py           # Data module init
│   └── datasets.py           # Data generation, loading, preprocessing
├── adapters/
│   ├── __init__.py
│   ├── timegan_adapter.py
│   ├── rgan_adapter.py
│   ├── gtgan_adapter.py
│   ├── timevae_adapter.py
│   ├── fourierflows_adapter.py
│   ├── csdi_adapter.py
│   ├── tsdiff_adapter.py
│   ├── tsdiff_patch.py       # GluonTS compatibility patch
│   └── diffusionts_adapter.py
├── repos/                    # Cloned method repositories (gitignored)
├── miniconda3/               # Conda installation (gitignored)
├── scripts/                  # Setup scripts
├── envs/                     # Environment configs
├── results/                  # Benchmark output (gitignored)
├── docs/
│   ├── metrics.md            # Detailed metric documentation
│   ├── datasets.md           # Dataset generation details
│   └── protocol.md           # Experimental protocol
├── KNOWN_ISSUES.md           # Known failures and workarounds
├── requirements.txt          # Python dependencies
└── verification_report.md    # Code audit findings
```

## References

- [Metrics documentation](docs/metrics.md) — detailed explanation of all Tier-A metrics
- [Datasets documentation](docs/datasets.md) — dataset generation and preprocessing
- [Protocol documentation](docs/protocol.md) — experimental protocol
- [Known Issues](KNOWN_ISSUES.md) — method failures and workarounds
- [Verification Report](verification_report.md) — code audit with major, medium, and low severity findings
