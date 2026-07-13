# Datasets

This document describes the generation, loading, and preprocessing of all 5 datasets used in the TSG SOTA Benchmark.

---

## 1. Sines

**File:** `data/datasets.py`, function `generate_sines()`

**Type:** Synthetic, 5 independent features

**Generation:**

$$x_{i,d}(t) = \sin(2\pi \cdot f_{i,d} \cdot t + \phi_{i,d})$$

- Frequency: $f_{i,d} \sim U(0, 0.1)$ (low-frequency sine waves)
- Phase: $\phi_{i,d} \sim U(0, 2\pi)$
- Each sample has 5 independent features with different random frequencies and phases
- Normalized to $[0, 1]$ via $(x + 1) \times 0.5$
- 10,000 samples per seed

**Purpose:** Simple controlled baseline with known generative structure. Tests basic unconditional generation capability.

---

## 2. Stocks

**File:** `data/datasets.py`, function `load_stock_data()`

**Type:** Real-world (Google stock data)

**Source:** TimeGAN preprocessing protocol

**Data:**
- Google (GOOG) daily OHLCV data (Open, High, Low, Close, Adj_Close)
- 5 feature columns
- Downloaded as part of the [TimeGAN repository](https://github.com/jsyoon0823/TimeGAN) (`repos/TimeGAN/data/stock_data.csv`)

**Preprocessing:**
1. Load first 5 columns (Open, High, Low, Close, Adj_Close)
2. Reverse the time series to chronological order
3. Create sliding windows with stride 1
4. Shuffle with fixed seed (seed=0) for deterministic ordering

**Purpose:** Standard benchmark for financial time series generation, widely used in the literature.

---

## 3. Energy

**File:** `data/datasets.py`, function `load_energy_data()`

**Type:** Real-world (UCI Appliances Energy Prediction)

**Source:** TimeGAN preprocessing protocol

**Data:**
- UCI Appliances energy prediction dataset
- 28 feature columns (temperature, humidity, pressure, etc. from sensors and weather station)
- Downloaded as part of the TimeGAN repository (`repos/TimeGAN/data/energy_data.csv`)

**Preprocessing:**
1. Load all 28 columns
2. Reverse to chronological order
3. Create sliding windows with stride 1
4. Shuffle with fixed seed (seed=0) for deterministic ordering

**Purpose:** High-dimensional multivariate time series benchmark. Tests a method's ability to capture cross-feature correlations.

---

## 4. Heston Stochastic Volatility

**File:** `data/datasets.py`, function `generate_heston()`

**Type:** Synthetic (stochastic volatility model)

**Model parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| mu | 0.05 | Drift (annualized) |
| kappa | 2.0 | Mean reversion speed |
| theta | 0.04 | Long-term variance |
| xi | 0.3 | Volatility of volatility |
| rho | -0.7 | Correlation (S, v) |
| S0 | 100.0 | Initial price |
| v0 | 0.04 | Initial variance |
| dt | 1/250 | Daily time step |

**Feller condition:** $2\kappa\theta = 0.16 \ge \xi^2 = 0.09$ -- satisfied, ensuring the CIR variance process does not hit zero in continuous time.

**Numerical scheme:** Euler-Maruyama with full truncation:

$$
\begin{aligned}
v_{t+1} &= v_t + \kappa(\theta - v_t^+) \Delta t + \xi \sqrt{v_t^+} \sqrt{\Delta t} \, Z_v \\
v_{t+1} &= \max(v_{t+1}, 0) \quad \text{(truncation)} \\
S_{t+1} &= S_t + \mu S_t \Delta t + \sqrt{v_t^+} S_t \sqrt{\Delta t} \, Z_s
\end{aligned}
$$

where $v_t^+ = \max(v_t, 0)$ and $Z_s, Z_v$ are correlated Brownian increments with correlation rho.

**Output:**
- Price paths S(t): shape (N, seq_len) -- the observed feature (1-D)
- Latent variance v(t): shape (N, seq_len) -- NOT given to models, used only for Teacher-Sigma metrics (A15)
- 10,000 samples per seed

**Purpose:** Tests whether a generative method can recover latent volatility dynamics (a hidden Markov structure) from observed price paths alone.

---

## 5. Sinusoidal Mixture

**File:** `data/datasets.py`, function `generate_sinusoidal_mixture()`

**Type:** Synthetic (multi-frequency mixture with controlled noise)

**Generation:**

$$X(t) = \sum_{k=1}^{K} A_k \cdot \sin(2\pi f_k t + \phi_k) + \varepsilon(t)$$

where:
- $A_k \sim U(0.5, 1.5)$ -- random amplitudes
- $f_k \sim U(0.01, 0.2)$ -- random frequencies
- $\phi_k \sim U(0, 2\pi)$ -- random phases
- $\varepsilon(t) \sim N(0, \sigma^2)$ -- Gaussian noise

**Signal-to-noise ratio:**

$$\text{SNR}_{\text{dB}} = 10 \log_{10} \left( \frac{\sigma_{\text{signal}}^2}{\sigma_{\text{noise}}^2} \right)$$

Noise variance is calibrated globally (same sigma for all samples) to achieve the target SNR.

**Combinations tested in the benchmark:**
- K = 3 components, SNR = 20 dB (default)
- Additional combinations K in {2, 3, 5} and SNR in {10 dB, 20 dB, inf} available via function parameters

**Output:** Shape (N, seq_len, 1) -- single-channel time series. 5,000 samples per combination.

**Purpose:** Controlled test of a method's ability to handle multi-scale temporal structure and noise robustness.

---

## Preprocessing

All datasets undergo the same preprocessing pipeline, applied identically across all methods for fair comparison.

### Normalization

Per-feature **min-max normalization** to $[0, 1]$:

$$x_{\text{norm}} = \frac{x - \text{min}}{\text{max} - \text{min} + 10^{-16}}$$

- Fit statistics (min, max) are computed **only on the training set** to avoid data leakage
- Test and generated data use the same training-set statistics for inverse transformation
- The inverse transform is applied to generated samples before computing metrics

### Train/Test Split

- Fixed 80/20 split using a deterministic permutation:
  ```python
  rng = np.random.default_rng(seed)
  idx = rng.permutation(N)
  ```
- Same seed always produces the same split, ensuring every method trains on the same data
- For Heston, the latent variance v(t) is split with the same permutation to maintain alignment

### Common Split Across All Methods

The `run_benchmark.py` script calls `load_dataset()` which loads the data fresh for each (dataset, seq_len, seed) combination and applies the same split using the same seed, guaranteeing all methods see identical training data.
