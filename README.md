# TSG SOTA Benchmark

Unified, reproducible benchmarking pipeline for time series generation methods.

## Methods

| # | Method | Family | Paper |
|---|--------|--------|-------|
| 1 | TimeGAN | GAN | Yoon et al., NeurIPS 2019 |
| 2 | RGAN / RCGAN | Recurrent GAN | Esteban et al., arXiv:1706.02633 |
| 3 | GT-GAN | GAN + Neural ODE/CDE | Jeon et al., NeurIPS 2022 |
| 4 | TimeVAE | VAE | Desai et al., arXiv:2111.08095 |
| 5 | Fourier Flows | Normalizing Flow | Alaa et al., ICLR 2021 |
| 6 | CSDI | Conditional Diffusion | Tashiro et al., NeurIPS 2021 |
| 7 | TSDiff | Unconditional Diffusion | Kollovieh et al., NeurIPS 2023 |
| 8 | Diffusion-TS | Interpretable Diffusion | Yuan & Qiao, ICLR 2024 |

## Datasets

1. **Sines** — Synthetic, 5 independent sine waves, 10k samples
2. **Stocks** — Google (GOOG) daily OHLCV, 5 features (TimeGAN protocol)
3. **Energy** — UCI Appliances energy prediction, 28 features (TimeGAN protocol)
4. **Heston** — Stochastic volatility (Euler-Maruyama, Feller-respecting), 1 feature + latent variance
5. **Sinusoidal mixture** — Multi-frequency sinusoids with controlled SNR

## Metrics (Tier A)

- Path MMD², Terminal MMD², Increment MMD²
- Volatility-discrepancy MMD
- Terminal / Path Sliced Wasserstein Distance
- Covariance Error, Mean RMSE
- Std Error, Kurtosis Error
- ACF Error (abs returns, squared returns)
- Discriminative Score (post-hoc GRU classifier)
- Predictive Score / TSTR (train on synthetic, test on real)
- Teacher-Sigma metrics (Heston only)

## Usage

```bash
# Small smoke test (5% data, 1 seed)
python run_benchmark.py --small

# Full sweep
python run_benchmark.py

# Subset
python run_benchmark.py --methods timegan,csdi --datasets sines,heston
```
