# Metrics — Tier-A Evaluation Suite

This document provides a detailed explanation of all 15 Tier-A metrics used in the TSG SOTA Benchmark. Metrics are computed by comparing generated (synthetic) time series against real (ground-truth) time series drawn from the same dataset's test split.

---

## A1: Path MMD^2 (Path Maximum Mean Discrepancy)

**File:** `metrics.py`, function `mmd2()`

Evaluates the discrepancy between the full joint distributions of generated and real time series paths. Uses a **multi-scale RBF (Gaussian) kernel** with bandwidth scales h in {0.5, 1.0, 2.0, 4.0, 8.0}:

$$k(u, v) = \frac{1}{L} \sum_{l=1}^{L} \exp\left(-\frac{\lVert u - v \rVert^2}{2 h_l^2 d}\right)$$

where d is the flattened dimensionality (T x n_features). Each sample (a path of length T) is flattened to a vector of length T x d before computing the kernel.

The MMD^2 estimator is:

$$\text{MMD}^2(X, Y) = \frac{1}{N_x^2} \sum_{i,j} k(x_i, x_j) + \frac{1}{N_y^2} \sum_{i,j} k(y_i, y_j) - \frac{2}{N_x N_y} \sum_{i,j} k(x_i, y_j)$$

Lower is better. This is the most comprehensive distributional metric as it captures the full path distribution including temporal dependencies.

---

## A2: Terminal MMD^2

**File:** `metrics.py`, function `terminal_mmd2()`

Same multi-scale RBF kernel as A1, but restricted to the **last time step** only. Evaluates how well the method captures the terminal distribution:

$$\text{MMD}^2(X[:, -1, :], Y[:, -1, :])$$

Particularly relevant for financial applications where the terminal value distribution matters (e.g., option pricing, risk management).

---

## A3: Increment MMD^2

**File:** `metrics.py`, function `increment_mmd2()`

Same multi-scale RBF kernel applied to the **first differences** (increments/returns):

$$dX_t = X_{t+1} - X_t \quad \text{for } t = 0, \dots, T-2$$

Measures whether the method captures the correct transition dynamics and short-term dependencies, independent of the level distribution.

---

## A4: Volatility MMD

**File:** `metrics.py`, function `volatility_mmd()`

A weighted sum of MMD^2 computed over 9 distinct stylized feature views of financial time series. Designed to capture whether the generated data reproduces known stylized facts of financial returns:

| View | Feature | Description |
|------|---------|-------------|
| 1 | Instantaneous RV | (dX_k)^2 / dt — raw squared increments |
| 2 | State-RV pairs | Concatenation of state X_{t+1} and RV_t |
| 3 | Global RV mean | Mean of realized variance per sample |
| 4 | Terminal return | X_T - X_0 |
| 5 | Returns (flattened) | All increments dX |
| 6 | Realized vol | Rolling window 5 std of squared returns |
| 7 | Absolute returns | |dX| |
| 8 | Squared returns | dX^2 |
| 9 | ACF lag products | Autocorrelation of |dX| and dX^2 at lags 1, 2, 5, 10 |

Each view produces a set of per-sample features. MMD^2 is computed independently for each view and summed to produce the final score.

---

## A5: Terminal Sliced Wasserstein Distance (Terminal SWD)

**File:** `metrics.py`, functions `slicing_wasserstein()` and `terminal_swd()`

Computes the **sliced 1-Wasserstein distance** between the terminal distributions:

$$\text{SWD}(X[:, -1, :], Y[:, -1, :]) = \frac{1}{N} \sum_{i=1}^{N} W_1(u_i^\top X, u_i^\top Y)$$

where u_i are N=50 random unit vectors drawn from a Gaussian and normalized, and W_1 is the 1-dimensional Wasserstein distance (equivalent to the L1 distance between quantile functions).

Sliced Wasserstein provides a tractable approximation of the full Wasserstein distance and is more sensitive to distributional differences than MMD in some settings.

---

## A6: Path SWD

**File:** `metrics.py`, function `path_swd()`

The mean of the Terminal SWD computed at every individual time step:

$$\text{Path SWD} = \frac{1}{T} \sum_{t=1}^{T} \text{SWD}(X[:, t, :], Y[:, t, :])$$

This measures distributional fidelity at each time step independently, providing a per-timestep assessment of quality. Unlike Path MMD^2, it does not capture cross-time dependencies.

---

## A7: Terminal Covariance Error

**File:** `metrics.py`, function `terminal_cov_error()`

Frobenius norm of the difference between the covariance matrices at the terminal time step:

$$||\text{Cov}(X_{gen}[:, -1, :]) - \text{Cov}(X_{real}[:, -1, :])||_F$$

Captures whether the model reproduces the correct feature correlations at the final time step.

---

## A8: Terminal Mean RMSE

**File:** `metrics.py`, function `terminal_mean_rmse()`

L2 norm of the difference between the mean vectors at the terminal time step:

$$||\mathbb{E}[X_{gen}[:, -1, :]] - \mathbb{E}[X_{real}[:, -1, :]]||_2$$

Measures bias in the terminal marginal distribution.

---

## A9: Return Standard Deviation Error

**File:** `metrics.py`, function `return_std_error()`

Absolute difference between the standard deviation of the increments (returns):

$$|\text{Std}(dX_{gen}) - \text{Std}(dX_{real})|$$

Measures whether the method generates the correct overall volatility level.

---

## A10: Return Kurtosis Error

**File:** `metrics.py`, function `return_kurtosis_error()`

Absolute difference between the excess kurtosis of the increments (Fisher's definition, where a normal distribution has kurtosis 0):

$$|\text{Kurt}(dX_{gen}) - \text{Kurt}(dX_{real})|$$

Captures tail behavior. Many financial time series exhibit heavy tails (positive excess kurtosis), and a good generative model should reproduce this.

---

## A11: ACF Error (Absolute Returns)

**File:** `metrics.py`, functions `acf()` and `acf_error()`

Mean absolute difference between the autocorrelation functions of **absolute returns** (|dX|) at lags 1, 2, 5, and 10:

$$\frac{1}{L} \sum_{l \in \{1,2,5,10\}} \left| \text{ACF}_l(|dX_{gen}|) - \text{ACF}_l(|dX_{real}|) \right|$$

The ACF of absolute returns measures volatility clustering — a key stylized fact of financial time series. Values are computed per-sample, per-feature and then averaged.

---

## A12: ACF Error (Squared Returns)

**File:** `metrics.py`, functions `acf()` and `acf_error()`

Same as A11 but on **squared returns** (dX^2). Squared returns are another common proxy for volatility in financial econometrics and exhibit stronger autocorrelation than raw returns.

---

## A13: Discriminative Score

**File:** `metrics.py`, function `compute_discriminative_score()`

**Implementation:** Delegates to TimeGAN's `metrics.discriminative_metrics` module.

A post-hoc RNN classifier (GRU with 2 layers, 12 hidden units, trained for 10000 steps with batch size 128) is trained to distinguish real from generated samples. The data is split 50/50 for training and testing the classifier.

The score is defined as:

$$|0.5 - \text{test\_accuracy}|$$

A perfect score is 0.0, meaning the classifier cannot distinguish real from generated (test accuracy = 50%). Higher scores indicate detectable differences.

**Note:** This metric requires TensorFlow 1.x (the TimeGAN repo). If unavailable, it returns NaN.

---

## A14: Predictive Score / TSTR (Train on Synthetic, Test on Real)

**File:** `metrics.py`, function `compute_predictive_score()`

**Implementation:** Delegates to TimeGAN's `metrics.predictive_metrics` module.

A one-step-ahead predictor (GRU with 2 layers, 12 hidden units, trained for 10000 steps with batch size 128) is trained on the **synthetic** data and evaluated on the **real** data. The score is the mean absolute error (MAE) of the predictions.

A lower score means the synthetic data contains sufficient temporal structure to train a useful predictive model.

**Note:** This metric requires TensorFlow 1.x (the TimeGAN repo). If unavailable, it returns NaN.

---

## A15: Teacher-Sigma Metrics (Heston Only)

**File:** `metrics.py`, function `teacher_sigma_metrics()`

Two metrics that evaluate whether the generated data captures the latent volatility dynamics — applicable only to the Heston stochastic volatility dataset where the true latent variance v(t) is known.

**Teacher-Sigma Correlation:** Pearson correlation between the estimated volatility (rolling window-5 standard deviation of returns) and the true latent sqrt(v(t)).

**Teacher-Sigma RMSE:** Root mean squared error between the estimated volatility and true sqrt(v(t)).

Higher correlation and lower RMSE indicate better recovery of the latent volatility process.

---

## Tier B (Reserved)

The following metrics are designated as Tier B and are not computed for baseline methods. They are reserved for future model-specific evaluation:

| Metric | Purpose |
|--------|---------|
| AC Loss | Adversarial critic loss |
| R Loss | Reconstruction loss |
| Trajectory RMSE | RMSE of full generated trajectories under an oracle |
| Control RMSE | Downstream control task error |
| Cost Gap | Cost differential in downstream tasks |
| Control Correlation | Correlation in control settings |
| Route Fraction | Fraction of valid generated paths |
| Control Energy | Energy expenditure in control tasks |
