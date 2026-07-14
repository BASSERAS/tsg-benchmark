# Full Sweep Roadmap

Sines seq_len=24 **done** (8 methods, 5 seeds, 14 metrics). Remaining datasets:

## Priority 1: Heston (stochastic volatility)
- 3 seq_lens: {24, 64, 128}
- 5 seeds each
- Uses `generate_heston()` from `data/datasets.py`
- Includes Teacher-Sigma Corr and RMSE metrics
- Est. time: ~24h on 4×A100s

## Priority 2: Stocks (GOOG OHLCV)
- Fixed length (TimeGAN protocol)
- 5 seeds
- Uses `load_stock_data()` from `data/datasets.py`
- Est. time: ~8h

## Priority 3: Energy (UCI appliances)
- 28 features, fixed length
- 5 seeds
- Uses `load_energy_data()` from `data/datasets.py`
- Est. time: ~12h

## Priority 4: Sinusoidal Mixture
- K × SNR grid (9 combos)
- 5 seeds each
- Est. time: ~48h

## How to continue
```bash
conda activate common_pt
python run_benchmark.py --datasets heston --seq-len 24
# ... then 64, 128
```
