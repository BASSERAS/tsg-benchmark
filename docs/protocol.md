# Experimental Protocol

This document defines the standardized experimental protocol used in the TSG SOTA Benchmark.

---

## Grid Configuration

The benchmark evaluates all combinations of:

| Dimension | Values | Count |
|-----------|--------|-------|
| Methods | timegan, rgan, gtgan, timevae, fourierflows, csdi, tsdiff, diffusionts | 8 |
| Datasets | sines, stocks, energy, heston, sinusoidal | 5 |
| Sequence lengths | 24, 64, 128 | 3 |
| Seeds | 0, 1, 2, 3, 4 | 5 |

**Total experiments:** 8 x 5 x 3 x 5 = **600** (in a full sweep)

Results are reported as mean +/- standard deviation across the 5 seeds.

---

## Training Budget

Training is constrained by a fixed number of **gradient steps**, not epochs. This ensures a fair comparison where each method receives the same optimization budget.

| Sequence Length | Training Steps | Batch Size |
|----------------|----------------|------------|
| 24 | 5,000 | 128 |
| 64 | 10,000 | 128 |
| 128 | 15,000 | 128 |

**Scaling rationale:** Longer sequences have more parameters to learn (more timesteps of dependencies), so the budget scales linearly with sequence length.

The training steps are passed to each adapter via the `training_steps` attribute, which the adapter must respect (either as iteration count or converted to an equivalent epoch budget).

### Per-Method Step Conversion

- **TimeGAN, RGAN** (TF 1.x): training_steps maps directly to `iterations` parameter.
- **GT-GAN** (PyTorch): training_steps is split between Phase 1 (embedding) and Phase 2 (adversarial). Phase 1 gets `min(10000, training_steps/2)` steps; Phase 2 gets the remainder.
- **TimeVAE** (TF 2.x): training_steps is converted to epochs: `epochs = max(1, training_steps * batch_size / n_samples)`.
- **Fourier Flows, TSDiff, Diffusion-TS** (PyTorch): training_steps maps directly to iteration/step count in the training loop.
- **CSDI** (PyTorch): training_steps is converted to epochs with the same formula as TimeVAE.

---

## Batch Configuration

- **Batch size:** 128 for all methods (default)
- In `--small` mode (5% data, used for validation), the effective dataset size may be smaller than the batch size. This is handled per-adapter (see KNOWN_ISSUES.md for known batch-size-related failures).

---

## Sampling

- **Samples per method:** `min(N_test, 2000)` -- up to 2000 samples, capped for memory
- All generated samples are subject to `np.clip(0, 1)` after generation (matching the normalization range)
- Generated data is inverse-transformed from normalized space to original scale before metric computation

---

## Hardware

| Component | Specification |
|-----------|--------------|
| GPUs | 4 x NVIDIA A100-SXM4-80GB |
| CPU | 2 x AMD EPYC 7763 (128 cores / 256 threads) |
| RAM | 503 GiB |
| CUDA | 8.0 |

**Parallelization:** Experiments are distributed across the 4 GPUs using round-robin assignment:

```python
gpu_id = gpu_ids[(experiment_index) % len(gpu_ids)]
```

Each GPU handles multiple experiments sequentially (one at a time). Jobs are not parallelized within a GPU. The benchmark does not use multi-GPU training for individual methods.

### Device Selection

- **PyTorch methods** (GT-GAN, CSDI, TSDiff, Diffusion-TS): Use `torch.cuda.is_available()` to detect GPUs. Device string is `cuda:{gpu_id}`.
- **Fourier Flows**: CPU only (uses numpy FFT internally). Device argument is overridden to CPU.
- **TensorFlow 1.x methods** (TimeGAN, RGAN): GPU selection is via `CUDA_VISIBLE_DEVICES` environment variable set by the runner.
- **TimeVAE** (TF 2.x): GPU selection follows TensorFlow's device placement.

---

## Metrics Computation

Metrics are computed on the **original-scale** data (after inverse normalization). The computation pipeline is:

1. Load real test data (N_test samples)
2. Generate N_samples <= min(N_test, 2000) synthetic samples
3. Inverse-transform generated samples to original scale
4. Compute all 15 Tier-A metrics comparing synthetic vs real
5. For Heston only: additionally compute Teacher-Sigma metrics using true latent variance

### Metrics Learned from Validation Process

During the development of this benchmark, the following protocol decisions were validated:

1. **TF1 environment for A13/A14:** The Discriminative Score (A13) and Predictive Score (A14) require TensorFlow 1.x from the TimeGAN repository. When running from the PyTorch environment (common_pt), these metrics return NaN. The validation on Sines (seq_len=24, 5 seeds) was completed for all 7 working methods using the tf1_env post-processing workaround. Results are in `results/tf1_sines_results.json`.

2. **Per-sample ACF computation:** A known bug (see verification_report.md) caused A11/A12 metrics to compute autocorrelation across sample boundaries. The current code uses the 3D (per-sample, per-feature) code path, which correctly averages ACF within each time series.

3. **Clipping to [0,1] is standard behavior:** All methods produce outputs clipped to [0,1] after generation, matching the normalization range. This is standard practice in the TimeGAN literature.

### Per-Seed Aggregation

Results are aggregated across 5 seeds:

```
mean +/- std = (1/5) * sum(metric_s) +/- sqrt((1/4) * sum(metric_s - mean)^2)
```

Tables report mean +/- std per (dataset, seq_len) group.

---

## Failure Handling

**Per-cell try/except:** Each experiment (one method x dataset x seq_len x seed) is wrapped in try/except blocks:

1. **Training failure:** If `adapter.fit()` raises an exception, the experiment is marked `FAILED` with the error message. Training time is not recorded.
2. **Sampling failure:** If `adapter.sample()` raises an exception, the experiment is marked `FAILED`.
3. **Metrics failure:** If metric computation fails, the metrics dict is replaced with an error entry. The experiment itself is still marked `OK`.

Failed experiments do not crash the sweep. They are logged in `results/failures.csv` and `results/failures.md`.

**Note:** The aggregate tables only include experiments with status `OK`. Failed experiments are reported separately.

---

## Data Handling

### Normalization

All data undergoes per-feature min-max normalization to [0, 1] using training-set statistics:

```python
train_norm = (train - min_vals) / (max_vals - min_vals + 1e-16)
test_norm = (test - min_vals) / (max_vals - min_vals + 1e-16)
```

### Split Consistency

The same 80/20 train/test split is used for all methods. The split is deterministic using the seed:

```python
rng = np.random.default_rng(seed)
idx = rng.permutation(N)
```

Each (dataset, seq_len, seed) combination produces the same data split every time, guaranteeing method-level fairness.

### Known Bug: Stock/Energy Double Normalization

The stock and energy datasets are normalized twice (once internally in `load_stock_data()` / `load_energy_data()`, and once by `min_max_normalize()` in `run_benchmark.py`). After the first pass, values are already in [0, 1]; the second pass produces values outside [0, 1]. This is documented in KNOWN_ISSUES.md.

---

## Output

Results are saved to the `results/` directory:

- `all_results.jsonl` -- Raw per-seed experiment results (appended during the sweep)
- `{dataset}_{seq_len}.csv` -- Per-method table with mean +/- std
- `{dataset}_{seq_len}.md` -- Markdown version of the table
- `aggregate_ranking.csv` / `aggregate_ranking.md` -- Ranking across all methods
- `failures.csv` / `failures.md` -- Failed experiments

The aggregate tables are only computed in non-small mode (full sweep).

---

## Small Mode (Smoke Test)

For validation, use `--small` (or `--dry-run`):

```
python run_benchmark.py --small --methods timegan,csdi --datasets sines --seq-len 24
```

- 5% of training data (minimum 32 samples)
- 1 seed (seed=0)
- Any subset of methods and datasets
- Results printed to console; no aggregate tables are saved

This is useful for verifying that all adapters load, train without errors, and produce samples of the correct shape.

---

## Validation Results

The benchmark was validated on Sines (seq_len=24) with all 7 non-failed methods across 5 seeds:

| Method | Status | Notes |
|--------|--------|-------|
| TimeGAN | OK | A13/A14 computed in tf1_env |
| RGAN | OK | A13/A14 computed in tf1_env |
| GT-GAN | OK | High kurtosis (known issue) |
| TimeVAE | OK | A13/A14 computed in tf1_env |
| Fourier Flows | OK | A13/A14 computed in tf1_env |
| CSDI | OK | A13/A14 computed in tf1_env |
| TSDiff | FAILED | Mode collapse (see KNOWN_ISSUES.md) |
| Diffusion-TS | OK | A13/A14 computed in tf1_env |

Full sweep results for Sines (seq_len=24) are available in `results/full_sines_results.json`. These include all 15 Tier-A metrics for all 7 working methods.
