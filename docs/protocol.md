# Experimental Protocol

## Scope

| Dimension | Value |
|-----------|-------|
| Methods | 8 (TimeGAN, RGAN, GT-GAN, TimeVAE, Fourier-flows, CSDI, TSDiff, Diffusion-TS) |
| Datasets | 5 (Sines, Stocks, Energy, Heston, Sinusoidal-mixture grid) |
| Sequence lengths | 24, 64, 128 |
| Seeds | 0, 1, 2, 3, 4 |
| Total experiments | 8 × (5 + 9 sinusoidal) × 3 × 5 ≈ 1,680 |

## Training Budget

Fixed gradient steps per (dataset, seq_len) — same budget for all methods:

| Seq Len | Gradient Steps | Batch Size |
|---------|---------------|------------|
| 24 | 5,000 | 128 |
| 64 | 10,000 | 128 |
| 128 | 15,000 | 128 |

No per-method hyperparameter tuning beyond setting n_features/seq_len.
Each method uses its official repo's published defaults for the closest dataset shape.

## Reproducibility

- **Data split**: Same train/test split (seeded) reused across all 8 methods and all 5 seeds
- **Model stochasticity**: Only the model's initialization/training varies across seeds
- **Evaluation**: All metrics computed by a single shared implementation (metrics.py)
- **Discriminative/Predictive scores**: Single shared GRU evaluator from TimeGAN's repo

## Hardware

| Resource | Specification |
|----------|-------------|
| GPU | 4× NVIDIA A100-SXM4-80GB |
| CPU | 128-core AMD EPYC 7763 |
| RAM | 503 GB |
| OS | Ubuntu 24.04 (Noble) |
| CUDA | 13.3 |

## Parallelization

Experiments are parallelized across the 4 GPUs by (method, dataset). 
Methods with conflicting dependency requirements use separate conda environments:

| Environment | Python | Methods |
|-------------|--------|---------|
| tf1_env | 3.7 | TimeGAN, RGAN |
| common_pt | 3.10 | GT-GAN, Fourier-flows, CSDI, TSDiff, Diffusion-TS |
| timevae_env | 3.12 | TimeVAE |

## Failure Handling

Each (method, dataset, seq_len, seed) cell is wrapped in try/except:
1. Training failure → caught, logged as FAILED, sweep continues
2. Sampling failure → caught, logged as FAILED
3. Metrics computation failure → caught, reported as NaN

## Output

- Per-(dataset, seq_len) tables in Markdown and CSV
- Each cell: mean ± std over 5 seeds
- Parameter counts, training time, peak GPU memory logged
- Aggregate ranking table across all metrics
- Failure report for any cells that errored
- Tier B metrics marked as N/A for all baselines
