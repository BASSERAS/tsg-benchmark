"""
TSG Benchmark — Data Generation and Preprocessing Module.

Provides synthetic data generators (sines, Heston, sinusoidal mixture),
real-data loaders (Google stock, UCI energy), normalization utilities,
sliding-window construction, and a high-level prepare_dataset() entry point.
"""

import os
import numpy as np
from typing import Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_TSG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEGAN_DATA = os.path.join(_TSG_ROOT, 'repos', 'TimeGAN', 'data')
_STOCK_PATH = os.path.join(_TIMEGAN_DATA, 'stock_data.csv')
_ENERGY_PATH = os.path.join(_TIMEGAN_DATA, 'energy_data.csv')


# ===================================================================
# 1. Synthetic data generators
# ===================================================================

def generate_sines(n_samples: int = 10000, seq_len: int = 24,
                   n_features: int = 5, seed: int = 0) -> np.ndarray:
    """
    x_{i,d}(t) = sin(2*pi*f_{i,d}*t + phi_{i,d})
    f_{i,d} ~ U(0, 0.1), phi_{i,d} ~ U(0, 2*pi)

    Parameters
    ----------
    n_samples : int
        Number of time series to generate.
    seq_len : int
        Length of each time series.
    n_features : int
        Number of feature dimensions.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Shape (n_samples, seq_len, n_features), values in [0, 1].
    """
    rng = np.random.default_rng(seed)

    # frequencies: (n_samples, n_features), phases: (n_samples, n_features)
    freqs = rng.uniform(0, 0.1, size=(n_samples, n_features))
    phases = rng.uniform(0, 2 * np.pi, size=(n_samples, n_features))

    # time grid: (seq_len,)
    t = np.arange(seq_len, dtype=np.float64)

    # data: sin(2*pi*f*t + phi) , shape (n_samples, seq_len, n_features)
    # Broadcasting: (n_samples, n_features, 1) * (seq_len,) -> (n_samples, n_features, seq_len)
    data = np.sin(2 * np.pi * freqs[:, :, None] * t[None, None, :]
                  + phases[:, :, None])
    # Transpose to (n_samples, seq_len, n_features)
    data = data.transpose(0, 2, 1)

    # Normalize to [0, 1] as in the original TimeGAN code
    data = (data + 1.0) * 0.5

    return data


def load_stock_data(seq_len: int = 24) -> np.ndarray:
    """
    Load Google stock data following the TimeGAN preprocessing protocol.

    Steps: reverse (to make chronological), per-feature min-max normalize,
    sliding windows with stride 1, shuffle.

    Uses columns Open, High, Low, Close, Adj_Close (first 5 columns).

    Parameters
    ----------
    seq_len : int
        Sliding window length.

    Returns
    -------
    np.ndarray
        Shape (N, seq_len, 5).
    """
    # Load first 5 columns: Open, High, Low, Close, Adj_Close
    raw = np.loadtxt(_STOCK_PATH, delimiter=',', skiprows=1,
                     usecols=(0, 1, 2, 3, 4))
    # Reverse to chronological order
    raw = raw[::-1]
    # Sliding windows
    data = create_windows(raw, seq_len, stride=1)
    # Shuffle
    rng = np.random.default_rng(0)
    rng.shuffle(data)
    return data


def load_energy_data(seq_len: int = 24) -> np.ndarray:
    """
    Load UCI energy data following the TimeGAN preprocessing protocol.

    Steps: reverse (to make chronological), per-feature min-max normalize,
    sliding windows with stride 1, shuffle.

    Parameters
    ----------
    seq_len : int
        Sliding window length.

    Returns
    -------
    np.ndarray
        Shape (N, seq_len, 28).
    """
    raw = np.loadtxt(_ENERGY_PATH, delimiter=',', skiprows=1)
    # Reverse to chronological order
    raw = raw[::-1]
    # Sliding windows
    data = create_windows(raw, seq_len, stride=1)
    # Shuffle
    rng = np.random.default_rng(0)
    rng.shuffle(data)
    return data


def generate_heston(n_samples: int = 10000, seq_len: int = 24,
                    seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Euler-Maruyama full-truncation scheme for the Heston stochastic
    volatility model.

    Parameters
    ----------
    n_samples : int
        Number of price paths.
    seq_len : int
        Number of time steps per path.
    seed : int
        Random seed.

    Returns
    -------
    S_paths : np.ndarray, shape (n_samples, seq_len)
        Simulated asset price. This is the observed feature (1-D).
    v_paths : np.ndarray, shape (n_samples, seq_len)
        Simulated latent variance. NOT given to models; used only for
        Teacher-Sigma metrics.
    """
    # Model parameters
    mu = 0.05
    kappa = 2.0
    theta = 0.04
    xi = 0.3
    rho = -0.7
    S0 = 100.0
    v0 = 0.04
    dt = 1.0 / 250.0

    rng = np.random.default_rng(seed)

    T = seq_len
    # Correlated Brownian increments
    z1 = rng.normal(size=(n_samples, T - 1))      # for S
    z2 = rng.normal(size=(n_samples, T - 1))      # independent component for v
    z_s = z1
    z_v = rho * z1 + np.sqrt(1.0 - rho ** 2) * z2

    sqrt_dt = np.sqrt(dt)

    S = np.empty((n_samples, T), dtype=np.float64)
    v = np.empty((n_samples, T), dtype=np.float64)
    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(1, T):
        v_plus = np.maximum(v[:, t - 1], 0.0)
        # Full-truncation: drift uses v_plus, diffusion uses sqrt(v_plus)
        v[:, t] = (v[:, t - 1]
                   + kappa * (theta - v_plus) * dt
                   + xi * np.sqrt(v_plus) * sqrt_dt * z_v[:, t - 1])
        v[:, t] = np.maximum(v[:, t], 0.0)          # truncate

        S[:, t] = (S[:, t - 1]
                   + mu * S[:, t - 1] * dt
                   + np.sqrt(v_plus) * S[:, t - 1] * sqrt_dt * z_s[:, t - 1])

    return S, v


def generate_sinusoidal_mixture(n_samples: int = 5000, seq_len: int = 24,
                                 K: int = 3, snr_db: float = 20.0,
                                 seed: int = 0) -> np.ndarray:
    """
    X(t) = sum_{k=1}^{K} A_k * sin(2*pi*f_k*t + phi_k) + eps(t)

    A_k ~ U(0.5, 1.5), f_k ~ U(0.01, 0.2), phi_k ~ U(0, 2*pi)
    eps(t) ~ N(0, sigma^2) where sigma is calibrated to achieve snr_db.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    seq_len : int
        Sequence length.
    K : int
        Number of sinusoidal components.
    snr_db : float
        Signal-to-noise ratio in dB. Use float('inf') for noiseless.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Shape (n_samples, seq_len, 1) — single-channel time series.
    """
    rng = np.random.default_rng(seed)

    # Random amplitudes, frequencies, phases
    A = rng.uniform(0.5, 1.5, size=(n_samples, K))       # (N, K)
    freqs = rng.uniform(0.01, 0.2, size=(n_samples, K))  # (N, K)
    phases = rng.uniform(0, 2 * np.pi, size=(n_samples, K))

    t = np.arange(seq_len, dtype=np.float64)  # (seq_len,)

    # Clean signal: (n_samples, K, seq_len) -> sum over K -> (n_samples, seq_len)
    clean = np.sum(A[:, :, None] * np.sin(2 * np.pi * freqs[:, :, None]
                                          * t[None, None, :]
                                          + phases[:, :, None]), axis=1)

    # Calibrate noise std from SNR
    if np.isinf(snr_db):
        sigma = 0.0
    else:
        # signal_power = variance of clean signal per sample -> mean across samples
        # We calibrate globally: sigma is constant for all samples
        signal_power = np.var(clean)          # scalar
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        sigma = np.sqrt(max(noise_power, 1e-16))

    noise = rng.normal(0, sigma, size=(n_samples, seq_len))
    data = clean + noise

    # Reshape to (n_samples, seq_len, 1)
    return data[:, :, None]


# ===================================================================
# 2. Normalization utilities
# ===================================================================

def _min_max_scaler(data: np.ndarray) -> np.ndarray:
    """Internal per-feature min-max scaler (fit and transform on same data)."""
    mn = data.min(axis=0)
    mx = data.max(axis=0)
    denom = mx - mn
    denom[denom == 0] = 1.0        # avoid division by zero
    return (data - mn) / (denom + 1e-16)


def min_max_normalize(train: np.ndarray, test: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-feature min-max normalize to [0, 1]. Fit on train only.

    Parameters
    ----------
    train : np.ndarray, shape (N_train, seq_len, n_features)
    test : np.ndarray, shape (N_test, seq_len, n_features)

    Returns
    -------
    train_norm : np.ndarray
    test_norm : np.ndarray
    min_vals : np.ndarray, shape (n_features,)
    max_vals : np.ndarray, shape (n_features,)
    """
    # Flatten over N and time to compute per-feature stats
    train_2d = train.reshape(-1, train.shape[-1])

    min_vals = train_2d.min(axis=0)
    max_vals = train_2d.max(axis=0)
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0

    train_norm = (train - min_vals) / (denom + 1e-16)
    test_norm = (test - min_vals) / (denom + 1e-16)
    return train_norm, test_norm, min_vals, max_vals


def inverse_min_max(norm_data: np.ndarray, min_vals: np.ndarray,
                    max_vals: np.ndarray) -> np.ndarray:
    """
    Inverse transform normalised data back to original scale.

    Parameters
    ----------
    norm_data : np.ndarray
    min_vals : np.ndarray, shape (n_features,)
    max_vals : np.ndarray, shape (n_features,)

    Returns
    -------
    np.ndarray
    """
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0
    return norm_data * (denom + 1e-16) + min_vals


# ===================================================================
# 3. Windowing and splitting
# ===================================================================

def create_windows(data: np.ndarray, seq_len: int,
                   stride: int = 1) -> np.ndarray:
    """
    Create sliding windows from a raw time series.

    Parameters
    ----------
    data : np.ndarray, shape (T, n_features)
        Univariate or multivariate time series.
    seq_len : int
        Window length.
    stride : int
        Step size between consecutive windows.

    Returns
    -------
    np.ndarray, shape (N_windows, seq_len, n_features)
    """
    T = data.shape[0]
    n_windows = (T - seq_len) // stride + 1
    if n_windows <= 0:
        raise ValueError(
            f"Time series length {T} is shorter than sequence length {seq_len}."
        )
    # sliding_window_view(axis=0) returns (N_windows, d, window_size).
    # Transpose to (N_windows, window_size, d) as expected downstream.
    windows = np.lib.stride_tricks.sliding_window_view(
        data, window_shape=(seq_len,), axis=0,
    )[::stride]
    windows = np.transpose(windows, (0, 2, 1))
    return np.ascontiguousarray(windows)


def train_test_split(data: np.ndarray, train_ratio: float = 0.8,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffle and split data into training and test sets.

    Same seed always produces the same split, ensuring fair comparison
    across methods.

    Parameters
    ----------
    data : np.ndarray, shape (N, ...)
    train_ratio : float
        Fraction of samples assigned to training.
    seed : int
        Random seed.

    Returns
    -------
    train : np.ndarray
    test : np.ndarray
    """
    N = data.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    split = int(N * train_ratio)
    train_idx = idx[:split]
    test_idx = idx[split:]
    return data[train_idx], data[test_idx]


# ===================================================================
# 4. High-level entry point
# ===================================================================

def prepare_dataset(name: str, seq_len: int = 24, seed: int = 0,
                    data_dir: Optional[str] = None) -> Dict:
    """
    Generate (or load), normalise, and split a dataset.

    Supported dataset names:

    - ``'sines'``         : synthetic sine data
    - ``'stock'``         : Google stock (TimeGAN)
    - ``'energy'``        : UCI energy (TimeGAN)
    - ``'heston'``        : Heston stochastic volatility
    - ``'sinusoidal_mixture'`` : noisy sinusoidal mixture (snr_db=20)
    - ``'sinusoidal_mixture_clean'`` : noiseless (snr_db=inf)

    Returns
    -------
    dict with keys:
        train, test            : normalised arrays (model input)
        train_raw, test_raw    : arrays in original scale
        min_vals, max_vals     : per-feature normalisation bounds
        n_features             : feature dimensionality
        n_train, n_test        : sample counts
        heston_v               : latent variance (only for ``'heston'``)
        (plus metadata specific to each dataset)
    """
    # ------------------------------------------------------------------
    # 1.  Generate or load the raw data
    # ------------------------------------------------------------------
    if name == 'sines':
        data = generate_sines(n_samples=10000, seq_len=seq_len, seed=seed)
    elif name == 'stock':
        data = load_stock_data(seq_len=seq_len)
    elif name == 'energy':
        data = load_energy_data(seq_len=seq_len)
    elif name == 'heston':
        S, v = generate_heston(n_samples=10000, seq_len=seq_len, seed=seed)
        # S has shape (N, seq_len) -> expand to (N, seq_len, 1)
        data = S[:, :, None]
    elif name in ('sinusoidal_mixture', 'sinusoidal_mixture_clean'):
        snr = float('inf') if name == 'sinusoidal_mixture_clean' else 20.0
        data = generate_sinusoidal_mixture(
            n_samples=5000, seq_len=seq_len, K=3, snr_db=snr, seed=seed,
        )
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    n_features = data.shape[-1]

    # ------------------------------------------------------------------
    # 2.  Split into train / test
    # ------------------------------------------------------------------
    train_raw, test_raw = train_test_split(data, train_ratio=0.8, seed=seed)

    # ------------------------------------------------------------------
    # 3.  Min-max normalise (fit on train only)
    # ------------------------------------------------------------------
    train, test, min_vals, max_vals = min_max_normalize(train_raw, test_raw)

    # ------------------------------------------------------------------
    # 4.  Build result dict
    # ------------------------------------------------------------------
    result = {
        'train': train,
        'test': test,
        'train_raw': train_raw,
        'test_raw': test_raw,
        'min_vals': min_vals,
        'max_vals': max_vals,
        'n_features': n_features,
        'n_train': train.shape[0],
        'n_test': test.shape[0],
        'heston_v': None,
    }

    if name == 'heston':
        # Split v the same way as S — use the same permutation
        N = v.shape[0]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        split = int(N * 0.8)
        result['train_raw'] = S[idx[:split], :, None]
        result['test_raw'] = S[idx[split:], :, None]
        result['train'] = (S[idx[:split], :, None] - min_vals) / (max_vals - min_vals + 1e-16)
        result['test'] = (S[idx[split:], :, None] - min_vals) / (max_vals - min_vals + 1e-16)
        result['heston_v'] = {
            'train': v[idx[:split]],
            'test': v[idx[split:]],
        }

    return result
