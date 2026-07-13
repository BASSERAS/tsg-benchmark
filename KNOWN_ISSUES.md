# Known Issues

## TSDiff — Mode Collapse (FAILED)

**Status**: ❌ FAILED across all datasets and seeds.

**Root cause**: GluonTS `context_length=0` incompatibility.
TSDiff's codebase uses GluonTS data format internally via `_extract_features()`, which
requires `context_length > 0` to split data into prior and context windows. When set to 0
(for unconditional full-sequence generation), the feature extraction breaks silently.

**Error mode**: All generated samples collapse to identical all-zero values after 
`np.clip(result, 0.0, 1.0)`. Global variance = 0.0.

**Attempted fixes**:
- Value clamping in `p_sample` loop → prevented NaN, caused mode collapse
- DDIM sampling → same zero-variance result
- `nan_to_num` in reverse diffusion → NaN fixed, still mode-collapsed
- Raw tensor `training_step` (bypassing GluonTS dict) → `UnboundLocalError: features`

**Workaround**: None found. The model needs proper GluonTS dataset integration with
`context_length > 0` in a forecasting setup. Unconditional full-sequence generation
is not supported by the current adapter.

**Benchmark impact**: All TSDiff cells read `FAILED` in the result tables.
Time spent on TSDiff is excluded from method timing comparisons.

## Fourier-flows — NaN in Spectral Normalization (RESOLVED)

**Status**: ✅ Fixed.

**Root cause**: Zero-variance spectral bin at the Nyquist frequency. For real-valued 
signals, the imaginary component at the highest frequency bin is identically zero,
causing `fft_std = 0` and `(x - mean) / std → NaN`.

**Fix**: Added `+ 1e-6` to `fft_std` in spectral normalization (line 190).

## TF1 Compatibility Warnings

**Status**: ✅ Cosmetic only.

The TimeGAN and RGAN adapters use `tf.compat.v1` mode on TensorFlow 2.x. TF prints
deprecation warnings about `tf.placeholder`, `tf.Session`, etc. These are non-fatal.

## GPU Memory on Shared Machines

**Status**: ⚠️ Informational.

The benchmark expects 4× A100-80GB GPUs. On shared machines, check GPU availability
with `nvidia-smi` before launching. If GPUs are occupied, set `CUDA_VISIBLE_DEVICES`
or use CPU fallback (much slower).
