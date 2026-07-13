# Datasets

Five datasets are used in this benchmark. Each produces `(N, seq_len, n_features)` arrays.

## 1. Sines (Synthetic)

- **Formula**: `x_{i,d}(t) = sin(2π · f_{i,d} · t + φ_{i,d})`
- **Parameters**: `f_{i,d} ~ U(0, 0.1)`, `φ_{i,d} ~ U(0, 2π)`, `t = 0, ..., seq_len-1`
- **Samples**: 10,000
- **Features**: 5 (independent)
- **Purpose**: Controlled sanity check with known ground truth

## 2. Stocks (Real)

- **Source**: Google (GOOG) daily OHLCV data, same download as TimeGAN's `data_loading.py`
- **Features**: 5 (Open, High, Low, Close, Volume)
- **Protocol**: TimeGAN's original preprocessing (reverse chronological, min-max normalize, sliding windows stride 1)
- **Purpose**: Real financial data benchmark

## 3. Energy (Real)

- **Source**: UCI "Appliances energy prediction" dataset
- **Features**: 28 (temperature, humidity, pressure, etc.)
- **Protocol**: Same as TimeGAN's original preprocessing
- **Purpose**: High-dimensional real data benchmark

## 4. Heston Stochastic Volatility (Synthetic)

- **Model**: Euler-Maruyama full-truncation scheme:
  ```
  S_{k+1} = S_k + μ · S_k · dt + √max(v_k,0) · S_k · √dt · Z1_k
  v_{k+1} = v_k + κ(θ - max(v_k,0)) · dt + ξ · √max(v_k,0) · √dt · Z2_k
  Z2_k = ρ · Z1_k + √(1-ρ²) · Z⟂_k
  ```
- **Calibration**: `μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=-0.7, S₀=100, v₀=0.04, dt=1/250`
- **Feller condition**: `2κθ = 0.16 ≥ ξ² = 0.09` (satisfied ✓)
- **Samples**: 10,000 at each seq_len {24, 64, 128}
- **Outputs**: S (price, fed to generators) + v (latent variance, only for Teacher-Sigma metrics)
- **Purpose**: Known ground-truth volatility for oracle evaluation

## 5. Sinusoidal Mixture (Synthetic)

- **Formula**: `X(t) = Σ_{k=1}^{K} A_k · sin(2π·f_k·t + φ_k) + ε(t)`
- **Parameters**: `A_k ~ U(0.5, 1.5)`, `f_k ~ U(0.01, 0.2)`, `φ_k ~ U(0, 2π)`
- **Grid**: K ∈ {2, 3, 5} × SNR ∈ {10dB, 20dB, ∞}
- **Samples**: 5,000 per (K, SNR) combination
- **Purpose**: Controlled frequency diversity + noise robustness

## Preprocessing (identical across all datasets)

1. Per-feature min-max normalize to [0, 1], fit on training split only
2. Sliding windows with stride 1 (for raw time series)
3. Shuffle windows, split 80% train / 20% test (same seed = same split across all methods)
4. Inverse-transform generated samples before computing metrics
