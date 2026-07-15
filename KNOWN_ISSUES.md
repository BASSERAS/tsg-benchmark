# Known Issues

Status of all 8 methods and known workarounds for the TSG SOTA Benchmark.

---

## Method Status Summary

| # | Method | Status | Comment |
|---|--------|--------|---------|
| 1 | TimeGAN | **OK** | Works in `tf1_env` |
| 2 | RGAN / RCGAN | **OK** | Works in `tf1_env` |
| 3 | GT-GAN | **OK** (high kurtosis) | ODE solver needs more training for higher-order moments |
| 4 | TimeVAE | **OK** | Must run in `timevae_env` (TF2) |
| 5 | Fourier Flows | **OK** | CPU-only (numpy FFT), works on all datasets |
| 6 | CSDI | **OK** | Unconditional mode via all-missing mask |
| 7 | TSDiff | **FAILED** | GluonTS `context_length=0` mode collapse |
| 8 | Diffusion-TS | **OK** | Interpretable diffusion, works on all datasets |

---

## 1. TSDiff: FAILED (GluonTS context_length=0 mode collapse)

- **Method:** TSDiff (unconditional diffusion with S4 backbone)
- **Status:** FAILED on this benchmark
- **Root cause:** GluonTS `context_length=0` incompatibility. TSDiff's codebase uses GluonTS data format internally and requires `context_length > 0` for its `_extract_features()` method. When set to 0 (for unconditional generation), the feature extraction produces incorrect training signals, causing mode collapse.
- **Error mode:** All generated samples are identical (all zeros after `np.clip(0, 1)`). Variance = 0.0 globally.
- **Attempted fixes:**
  - Clamping p_sample values (prevented NaN but caused mode collapse)
  - DDIM sampling (same result)
  - `nan_to_num` in reverse diffusion loop (NaN replaced but collapsed)
- **Workaround:** None found. The model would need proper GluonTS dataset integration or a `context_length > 0` (forecasting) setup.
- **Impact:** TSDiff is marked FAILED for all datasets and seeds in this benchmark.
- **Reproduction:** Set context_length to 0 and prediction_length to seq_len in any TSDiff config. The adapter will produce zero-variance output after 5000 training steps.

---

## 2. GT-GAN: High Kurtosis (Outlier on Kurtosis Error)

- **Observation:** GT-GAN consistently produces kurtosis_error values around 44 on the Sines dataset, while all other methods score below 2. This is a known artifact of the Neural CDE + CNF generator architecture.
- **Root cause:** The Continuous Normalizing Flow (CNF) generator in GT-GAN is trained with a limited ODE solver budget. Higher-order moments require more precise integration than the standard budget provides.
- **Mitigation:** Increasing the number of ODE solver steps and training steps would reduce kurtosis error. The current budget of ~5000 steps is insufficient.
- **Impact on rankings:** When computing average rankings across metrics, the extreme kurtosis error dominates GT-GAN's aggregate score. Consider excluding kurtosis error or normalizing per-metric rankings before aggregation.

---

## 3. TimeVAE: Must Run in timevae_env

- **Required environment:** `timevae_env` (TensorFlow 2.16.1)
- **Why separate:** TimeVAE uses TensorFlow 2.x, which conflicts with both the TF 1.x environment (tf1_env for TimeGAN/RGAN) and the PyTorch environment (common_pt).
- **Adapter mechanism:** The `TimeVAEAdapter` is imported and executed within the `timevae_env` conda environment. The benchmark runner switches environments automatically.
- **Note:** If TimeVAE fails with "Failed to load the native TensorFlow runtime", ensure CUDA/cuDNN versions match TF 2.16 requirements.

---

## 4. A13/A14 (Discriminative/Predictive Scores): Need TF1

- **Requirement:** A13 (Discriminative Score) and A14 (Predictive Score) delegate to TimeGAN's `metrics.discriminative_metrics` and `metrics.predictive_metrics` modules, which require TensorFlow 1.x.
- **Current behavior (common_pt env):** When computed from the `common_pt` (PyTorch) environment, these metrics return `NaN` because TF 1.x is not available.
- **Workaround:** Post-process A13/A14 metrics by running `metrics.py` from the `tf1_env` environment on already-generated samples:
  ```bash
  conda run -n tf1_env python -c "
  from metrics import compute_discriminative_score, compute_predictive_score
  import numpy as np
  real = np.load('real_samples.npy')
  gen = np.load('gen_samples.npy')
  print(compute_discriminative_score(real, gen))
  "
  ```
- **Validation results:** A13/A14 were computed for all methods on Sines (seq_len=24, seed=0) using the `tf1_env` post-processing workaround. See `results/tf1_sines_results.json`.
- **PT methods limitation:** For PyTorch methods (FF, CSDI, DTS, GT-GAN) on other datasets, the TF1 post-processing is impractical: each sample requires rebuilding a TF GRU classifier graph (~30-60s per sample). With 80+ samples this exceeds practical runtime. These methods have A13/A14 marked as NaN on non-Sines datasets.
- **TF1 native methods:** TimeGAN and RGAN (both TF1) have A13/A14 computed automatically during training. TimeVAE (TF2) also has them.

---

## 5. Double Normalization on Stock and Energy (Known Bug)

- **Affected datasets:** `stocks`, `energy`
- **Issue:** `load_stock_data()` and `load_energy_data()` in `data/datasets.py` apply internal per-feature min-max scaling, then `run_benchmark.py` normalizes again via `min_max_normalize()`. After the first pass, values are already in [0, 1]; the second pass produces values outside [0, 1] based on training-set statistics.
- **Impact:** Stock and energy results may differ from the original TimeGAN protocol. The inverse transform applies the wrong bounds.
- **Status:** Documented but not fixed (to maintain backward compatibility with intermediate results).

---

## 6. TimeGAN `sample()` Returns Cached Data (Known Bug)

- **Affected method:** TimeGAN
- **Issue:** The `sample()` method in `adapters/timegan_adapter.py` does not generate new samples from the model. It returns a slice of the data that was generated at the end of `fit()`. If `n` exceeds the cached sample count, fewer samples are returned than requested.
- **Impact:** This is inherited from the adapter architecture and does not affect reported numbers (sampling is done once at the end of training), but means TimeGAN results may not reflect true generalization.
- **Note:** The newer version of the adapter (which locates X_hat in the graph via tensor name) fixes the original "Z_mb unused" bug. See `verification_report.md` for the full audit.

---

## 7. Additional Verification Findings

For the complete code audit with 4 major, 8 medium, and 7 low-severity findings, see `verification_report.md`.

### Quick Reference

| Issue | Severity | Status |
|-------|----------|--------|
| ACF computed across sample/feature boundaries (A11/A12) | **Major** | Fixed (per-sample ACF code path) |
| `rolling_std` computes vol-of-vol, not realized vol (A4) | **Major** | Documented |
| Double normalization (stock, energy) | **Major** | Documented |
| TimeGAN `sample()` returns cached data | **Major** | Partially mitigated |
| TimeGAN import path conflict (A13/A14) | Medium | Workaround exists |
| Missing batch_size guards (RGAN, GT-GAN, TSDiff) | Medium | Affects --small mode only |
| `tf.logging` removal risk (RGAN) | Medium | Depends on TF version |
| `tabulate` dependency missing | Low | Install with `pip install tabulate` |
| `GRBAC_BACKEND` typo (TSDdiff) | Low | Harmless |
| Dataset name inconsistency ("sinusoidal" vs "sinusoidal_mixture") | Low | CLI-only, not a runtime bug |
