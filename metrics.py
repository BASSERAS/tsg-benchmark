"""
TSG Benchmark — Metrics Module.

Defines the Tier-A evaluation metrics for time-series generation quality:

    A1   Full joint-path MMD^2
    A2   Terminal MMD^2
    A3   Increment MMD^2
    A4   Volatility-discrepancy MMD
    A5   Terminal Sliced Wasserstein Distance
    A6   Path Sliced Wasserstein Distance
    A7   Terminal Covariance Error
    A8   Terminal Mean RMSE
    A9   Return Std Error
    A10  Return Kurtosis Error
    A11  ACF Error (absolute returns)
    A12  ACF Error (squared returns)
    A13  Discriminative Score          (TimeGAN impl.)
    A14  Predictive Score              (TimeGAN impl.)
    A15  Teacher-Sigma Correlation     (Heston only)
         Teacher-Sigma RMSE           (Heston only)
"""

import sys
import os
import importlib
import numpy as np
from scipy.stats import wasserstein_distance, kurtosis
from typing import Optional, Callable, Tuple, Dict, Any


# ===================================================================
# 1.  Kernel helpers
# ===================================================================

def rbf_multiscale_kernel(
    u: np.ndarray, v: np.ndarray,
    scales: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
) -> np.ndarray:
    r"""Multi-scale RBF (Gaussian) kernel.

    .. math::

        k(u, v) = \frac{1}{L} \sum_{l=1}^{L}
            \exp\left(-\frac{\lVert u - v \rVert^2}{2 h_l^2 d}\right)

    Parameters
    ----------
    u : np.ndarray, shape (N_u, d)
    v : np.ndarray, shape (N_v, d)
    scales : tuple of float
        Bandwidth scales :math:`h_l`.

    Returns
    -------
    np.ndarray, shape (N_u, N_v)
        Kernel matrix.
    """
    d = u.shape[-1]
    # squared pairwise Euclidean distances  (N_u, N_v)
    diff2 = np.sum((u[:, None, :] - v[None, :, :]) ** 2, axis=-1)
    result = np.zeros_like(diff2)
    for h in scales:
        result += np.exp(-diff2 / (2.0 * h ** 2 * d))
    return result / len(scales)


# ===================================================================
# 2.  MMD-based metrics (A1 -- A4)
# ===================================================================

def mmd2(
    X: np.ndarray, Y: np.ndarray,
    kernel: Callable = rbf_multiscale_kernel,
) -> float:
    r"""A1. Full joint-path MMD^2.

    .. math::

        \text{MMD}^2(X, Y) = \mathbb{E}[k(X, X')] + \mathbb{E}[k(Y, Y')]
            - 2 \mathbb{E}[k(X, Y)]

    Parameters
    ----------
    X : np.ndarray, shape (N_x, T, d)
    Y : np.ndarray, shape (N_y, T, d)
    kernel : callable
        Kernel function ``k(u, v) -> np.ndarray (N_u, N_v)``.

    Returns
    -------
    float
    """
    N_x = X.shape[0]
    N_y = Y.shape[0]

    # Flatten the time dimension so each sample is a vector of length T*d
    X_flat = X.reshape(N_x, -1)
    Y_flat = Y.reshape(N_y, -1)

    kxx = kernel(X_flat, X_flat)
    kyy = kernel(Y_flat, Y_flat)
    kxy = kernel(X_flat, Y_flat)

    # Biased estimator
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def terminal_mmd2(
    X: np.ndarray, Y: np.ndarray,
    kernel: Callable = rbf_multiscale_kernel,
) -> float:
    """A2. Terminal MMD^2 -- restricted to the last time step t = T."""
    return mmd2(X[:, -1:, :], Y[:, -1:, :], kernel)


def increment_mmd2(
    X: np.ndarray, Y: np.ndarray,
    kernel: Callable = rbf_multiscale_kernel,
) -> float:
    """A3. Increment MMD^2 on the first differences dX = X_{t+1} - X_t."""
    return mmd2(np.diff(X, axis=1), np.diff(Y, axis=1), kernel)


def volatility_mmd(
    X: np.ndarray, Y: np.ndarray,
    kernel: Callable = rbf_multiscale_kernel,
) -> float:
    r"""A4. Volatility-discrepancy MMD.

    Computes a weighted sum of MMD^2 over several stylised feature views
    of financial time series: instantaneous realised variance (RV),
    state-RV pairs, global RV mean, terminal return, returns, realised
    volatility (rolling window of 5), absolute returns, squared returns,
    and ACF lag-products of |dX| and dX^2 for lags {1, 2, 5, 10}.

    Parameters
    ----------
    X : np.ndarray, shape (N_x, T, d)
    Y : np.ndarray, shape (N_y, T, d)
    kernel : callable

    Returns
    -------
    float
    """
    dt = 1.0
    dX = np.diff(X, axis=1)
    dY = np.diff(Y, axis=1)

    features_X: list[np.ndarray] = []
    features_Y: list[np.ndarray] = []

    # --- 1. Instantaneous realised variance  RV_k = (dX_k)^2 / dt
    rv_X = (dX ** 2) / dt
    rv_Y = (dY ** 2) / dt
    features_X.append(rv_X.reshape(rv_X.shape[0], -1))
    features_Y.append(rv_Y.reshape(rv_Y.shape[0], -1))

    # --- 2. State-RV pairs  (X_{t+1}, RV_t)
    state_rv_X = np.concatenate([X[:, 1:, :], rv_X], axis=-1)
    state_rv_Y = np.concatenate([Y[:, 1:, :], rv_Y], axis=-1)
    features_X.append(state_rv_X.reshape(state_rv_X.shape[0], -1))
    features_Y.append(state_rv_Y.reshape(state_rv_Y.shape[0], -1))

    # --- 3. Global RV mean (across time) per sample
    features_X.append(np.mean(rv_X, axis=1))
    features_Y.append(np.mean(rv_Y, axis=1))

    # --- 4. Terminal return  X_T - X_0
    features_X.append(X[:, -1, :] - X[:, 0, :])
    features_Y.append(Y[:, -1, :] - Y[:, 0, :])

    # --- 5. Returns dX (flattened across time and features)
    features_X.append(dX.reshape(dX.shape[0], -1))
    features_Y.append(dY.reshape(dY.shape[0], -1))

    # --- 6. Realised volatility (rolling window of 5 on squared returns)
    def rolling_std(x_sq: np.ndarray, window: int = 5) -> np.ndarray:
        x_pad = np.pad(x_sq, ((0, 0), (window - 1, 0), (0, 0)),
                       mode='edge')
        out = np.zeros_like(x_sq)
        for i in range(x_sq.shape[1]):
            out[:, i, :] = np.mean(x_pad[:, i:i + window, :], axis=1)
        return np.sqrt(out ** 2 + 1e-6)

    rvol_X = rolling_std(dX ** 2)
    rvol_Y = rolling_std(dY ** 2)
    features_X.append(rvol_X.reshape(rvol_X.shape[0], -1))
    features_Y.append(rvol_Y.reshape(rvol_Y.shape[0], -1))

    # --- 7. Absolute returns
    features_X.append(np.abs(dX).reshape(dX.shape[0], -1))
    features_Y.append(np.abs(dY).reshape(dY.shape[0], -1))

    # --- 8. Squared returns
    features_X.append((dX ** 2).reshape(dX.shape[0], -1))
    features_Y.append((dY ** 2).reshape(dY.shape[0], -1))

    # --- 9. ACF lag-products
    for lag in [1, 2, 5, 10]:
        if lag >= dX.shape[1]:
            continue
        # ACF of |dX|
        acf_X = np.array(
            [acf(np.abs(dX[i, :, f]), lag)
             for i in range(dX.shape[0])
             for f in range(dX.shape[2])]
        )
        acf_Y = np.array(
            [acf(np.abs(dY[i, :, f]), lag)
             for i in range(dY.shape[0])
             for f in range(dY.shape[2])]
        )
        features_X.append(acf_X.reshape(-1, 1))
        features_Y.append(acf_Y.reshape(-1, 1))

        # ACF of dX^2
        acf_X_sq = np.array(
            [acf(dX[i, :, f] ** 2, lag)
             for i in range(dX.shape[0])
             for f in range(dX.shape[2])]
        )
        acf_Y_sq = np.array(
            [acf(dY[i, :, f] ** 2, lag)
             for i in range(dY.shape[0])
             for f in range(dY.shape[2])]
        )
        features_X.append(acf_X_sq.reshape(-1, 1))
        features_Y.append(acf_Y_sq.reshape(-1, 1))

    # Sum over all feature groups
    total = 0.0
    for fx, fy in zip(features_X, features_Y):
        total += mmd2(fx, fy, kernel)

    return total


# ===================================================================
# 3.  Sliced Wasserstein Distance  (A5 -- A6)
# ===================================================================

def slicing_wasserstein(
    X: np.ndarray, Y: np.ndarray, n_proj: int = 50, seed: int = 0,
) -> float:
    """Sliced Wasserstein Distance between two empirical distributions.

    Parameters
    ----------
    X : np.ndarray, shape (N_x, d)
    Y : np.ndarray, shape (N_y, d)
    n_proj : int
        Number of random projections.
    seed : int
        Random seed.

    Returns
    -------
    float
    """
    rng = np.random.default_rng(seed)
    d = X.shape[-1]
    dists = []
    for _ in range(n_proj):
        u = rng.normal(size=d)
        u /= np.linalg.norm(u) + 1e-16
        proj_X = X @ u
        proj_Y = Y @ u
        dists.append(wasserstein_distance(proj_X, proj_Y))
    return float(np.mean(dists))


def terminal_swd(
    X: np.ndarray, Y: np.ndarray, n_proj: int = 50, seed: int = 0,
) -> float:
    """A5. Terminal Sliced Wasserstein Distance (at t = t_N)."""
    return slicing_wasserstein(X[:, -1, :], Y[:, -1, :], n_proj, seed)


def path_swd(
    X: np.ndarray, Y: np.ndarray, n_proj: int = 50, seed: int = 0,
) -> float:
    """A6. Path SWD -- mean of Terminal-SWD at every time step."""
    T = X.shape[1]
    scores = []
    for t in range(T):
        scores.append(
            slicing_wasserstein(X[:, t, :], Y[:, t, :], n_proj, seed)
        )
    return float(np.mean(scores))


# ===================================================================
# 4.  Distributional moment metrics  (A7 -- A12)
# ===================================================================

def terminal_cov_error(X: np.ndarray, Y: np.ndarray) -> float:
    """A7. Terminal Covariance Error -- Frobenius norm of the covariance
    matrix difference at the final time step."""
    cov_X = np.atleast_2d(np.cov(X[:, -1, :].T))
    cov_Y = np.atleast_2d(np.cov(Y[:, -1, :].T))
    return float(np.linalg.norm(cov_X - cov_Y, ord='fro'))


def terminal_mean_rmse(X: np.ndarray, Y: np.ndarray) -> float:
    """A8. Terminal Mean RMSE -- RMSE between mean vectors at t = T."""
    return float(np.linalg.norm(
        X[:, -1, :].mean(0) - Y[:, -1, :].mean(0),
    ))


def return_std_error(X: np.ndarray, Y: np.ndarray) -> float:
    """A9. Return Std Error -- absolute difference of the standard
    deviations of the first differences."""
    dX = np.diff(X, axis=1)
    dY = np.diff(Y, axis=1)
    return float(abs(np.std(dX) - np.std(dY)))


def return_kurtosis_error(X: np.ndarray, Y: np.ndarray) -> float:
    """A10. Return Kurtosis Error -- absolute difference of the excess
    kurtosis of the first differences (using Fisher's definition, so
    normal distribution has kurtosis 0)."""
    dX = np.diff(X, axis=1).ravel()
    dY = np.diff(Y, axis=1).ravel()
    kx = kurtosis(dX, fisher=True, bias=False)
    ky = kurtosis(dY, fisher=True, bias=False)
    # Guard against NaN when the input is near-constant
    if np.isnan(kx) or np.isnan(ky):
        return 0.0
    return float(abs(kx - ky))


def acf(q: np.ndarray, lag: int) -> float:
    """Auto-correlation function at a given lag.

    Parameters
    ----------
    q : np.ndarray, shape (T,)
    lag : int

    Returns
    -------
    float
    """
    q = q - q.mean()
    denom = np.sum(q ** 2)
    if denom < 1e-16:
        return 0.0
    return float(np.sum(q[:-lag] * q[lag:]) / denom)


def acf_error(
    q_gen: np.ndarray, q_real: np.ndarray, lags=(1, 2, 5, 10),
) -> float:
    """A11 / A12. Mean absolute ACF error over multiple lags.

    Use with *absolute returns* for A11 and *squared returns* for A12.

    Parameters
    ----------
    q_gen : np.ndarray
        Time series data from generated data (e.g. |dX| or dX^2).
        If 3D (N, T, d), computes ACF per-sample per-feature then
        averages across lags.  If 1D (T,), computes directly.
    q_real : np.ndarray
        Corresponding reference data, same shape convention.
    lags : tuple of int

    Returns
    -------
    float
    """
    # 3D arrays (N, T, d): per-sample, per-feature ACF, then compare
    # mean ACF per lag and average across lags.
    if q_gen.ndim == 3 and q_real.ndim == 3:
        errors = []
        for lag in lags:
            if lag >= q_real.shape[1]:
                continue
            real_vals = np.array([
                acf(q_real[i, :, f], lag)
                for i in range(q_real.shape[0])
                for f in range(q_real.shape[2])
            ])
            gen_vals = np.array([
                acf(q_gen[i, :, f], lag)
                for i in range(q_gen.shape[0])
                for f in range(q_gen.shape[2])
            ])
            errors.append(abs(np.mean(real_vals) - np.mean(gen_vals)))
        if not errors:
            return 0.0
        return float(np.mean(errors))

    # 1D fallback (original behavior — backward compat)
    if np.std(q_real) < 1e-16 or np.std(q_gen) < 1e-16:
        return 0.0
    return float(np.mean(
        [abs(acf(q_gen, lag) - acf(q_real, lag)) for lag in lags]
    ))


# ===================================================================
# 5.  Teacher-Sigma metrics  (A15 -- Heston only)
# ===================================================================

def _rolling_mean_std_5(dX_sq: np.ndarray) -> np.ndarray:
    """Rolling window-5 mean of squared increments, returned as vol."""
    out = np.zeros_like(dX_sq)
    cum = np.cumsum(dX_sq, axis=1)
    for i in range(dX_sq.shape[1]):
        start = max(0, i - 4)
        w = i - start + 1
        out[:, i, :] = (cum[:, i, :] - (cum[:, start - 1, :]
                       if start > 0 else 0)) / w
    return np.sqrt(out + 1e-6)


def teacher_sigma_metrics(
    X_gen: np.ndarray, v_true: np.ndarray,
) -> Tuple[float, float]:
    """A15. Teacher-Sigma Correlation and RMSE (Heston model).

    Estimates latent volatility as the rolling window-5 standard
    deviation of returns and compares it against the true latent
    volatility sqrt(v_true).

    Parameters
    ----------
    X_gen : np.ndarray, shape (N, T, d)
        Generated price paths.
    v_true : np.ndarray, shape (N, T)
        True latent variance paths.

    Returns
    -------
    corr : float
        Pearson correlation between estimated and true volatility.
    rmse : float
        Root mean squared error between estimated and true volatility.
    """
    dX = np.diff(X_gen, axis=1)          # (N, T-1, d)
    sigma_hat = _rolling_mean_std_5(dX ** 2)   # (N, T-1, d)

    # v_true may have length T; the increments have T-1
    if v_true.shape[1] > dX.shape[1]:
        v_sqrt = np.sqrt(np.maximum(v_true[:, 1:], 0.0))
    else:
        v_sqrt = np.sqrt(np.maximum(v_true, 0.0))

    # Align shapes (squeeze feature dim if needed)
    sh = sigma_hat.ravel()
    vs = v_sqrt.ravel() if v_sqrt.ndim == 2 else v_sqrt[:, :, None].ravel()
    min_n = min(len(sh), len(vs))

    sh = sh[:min_n]
    vs = vs[:min_n]

    corr = np.corrcoef(sh, vs)[0, 1]
    rmse = np.sqrt(np.mean((sh - vs) ** 2))

    corr = float(corr) if not np.isnan(corr) else 0.0
    return corr, float(rmse)


# ===================================================================
# 6.  TimeGAN-based metrics  (A13 -- A14)
#     These import the original TimeGAN code at runtime.
# ===================================================================

def _import_timegan_metric(name: str) -> Any:
    """Dynamically import a metric function from the TimeGAN repo.

    Adds ``repos/TimeGAN`` to ``sys.path`` if it isn't already there.
    """
    timegan_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'repos', 'TimeGAN',
    )
    if timegan_dir not in sys.path:
        sys.path.insert(0, timegan_dir)

    spec = importlib.util.find_spec(f'metrics.{name}')
    if spec is None:
        raise ImportError(
            f"Could not find TimeGAN metrics module "
            f"``metrics.{name}`` at {timegan_dir}. "
            f"Ensure the TimeGAN repo is present."
        )

    mod = importlib.import_module(f'metrics.{name}')
    return mod


def _convert_to_list_of_arrays(data: np.ndarray) -> list:
    """Convert a batched ndarray (N, T, d) to a list of length N of
    (T, d) arrays, as expected by the TimeGAN interface."""
    return [data[i] for i in range(data.shape[0])]


def compute_discriminative_score(
    real_data: np.ndarray, gen_data: np.ndarray, seed: int = 0,
) -> float:
    """A13. Discriminative Score via TimeGAN's post-hoc RNN.

    A post-hoc RNN is trained to distinguish real from generated samples.
    The score is ``|0.5 - test_accuracy|``.

    Parameters
    ----------
    real_data : np.ndarray, shape (N, T, d)
    gen_data : np.ndarray, shape (N, T, d)
    seed : int
        Random seed (numpy seed is set for reproducibility).

    Returns
    -------
    float
    """
    np.random.seed(seed)

    metric_mod = _import_timegan_metric('discriminative_metrics')
    score = metric_mod.discriminative_score_metrics(
        _convert_to_list_of_arrays(real_data),
        _convert_to_list_of_arrays(gen_data),
    )
    return float(score)


def compute_predictive_score(
    real_data: np.ndarray, gen_data: np.ndarray, seed: int = 0,
) -> float:
    """A14. Predictive Score via TimeGAN's post-hoc RNN (TSTR).

    A one-step-ahead predictor is trained on the synthetic data and
    evaluated on the real data. Score is the mean absolute error.

    Parameters
    ----------
    real_data : np.ndarray, shape (N, T, d)
    gen_data : np.ndarray, shape (N, T, d)
    seed : int
        Random seed.

    Returns
    -------
    float
    """
    np.random.seed(seed)

    metric_mod = _import_timegan_metric('predictive_metrics')
    score = metric_mod.predictive_score_metrics(
        _convert_to_list_of_arrays(real_data),
        _convert_to_list_of_arrays(gen_data),
    )
    return float(score)


# ===================================================================
# 7.  Aggregate runner
# ===================================================================

def compute_all_metrics(
    X_gen: np.ndarray,
    X_real: np.ndarray,
    heston_v: Optional[np.ndarray] = None,
    seed: int = 0,
) -> dict:
    """Compute all Tier-A metrics.

    Parameters
    ----------
    X_gen : np.ndarray, shape (N_gen, T, d)
        Generated / synthetic time series.
    X_real : np.ndarray, shape (N_real, T, d)
        Real / ground-truth time series.
    heston_v : np.ndarray or None, shape (N_gen, T)
        True latent variance from the Heston generator. Only required
        for Teacher-Sigma metrics.
    seed : int
        Random seed for stochastic metrics (SWD and TimeGAN-based).

    Returns
    -------
    dict
        Metric name -> value.
    """
    results: Dict[str, float] = {}

    # --- MMD-based ---
    results['path_mmd2'] = mmd2(X_gen, X_real)
    results['terminal_mmd2'] = terminal_mmd2(X_gen, X_real)
    results['increment_mmd2'] = increment_mmd2(X_gen, X_real)
    results['volatility_mmd'] = volatility_mmd(X_gen, X_real)

    # --- SWD ---
    results['terminal_swd'] = terminal_swd(X_gen, X_real, seed=seed)
    results['path_swd'] = path_swd(X_gen, X_real, seed=seed)

    # --- Distributional moments ---
    results['cov_error'] = terminal_cov_error(X_gen, X_real)
    results['mean_rmse'] = terminal_mean_rmse(X_gen, X_real)
    results['std_error'] = return_std_error(X_gen, X_real)
    results['kurtosis_error'] = return_kurtosis_error(X_gen, X_real)

    # --- ACF errors ---
    # Per-sample, per-feature ACF; average across lags.
    dX = np.diff(X_real, axis=1)          # (N_real, T-1, d)
    dY = np.diff(X_gen, axis=1)           # (N_gen, T-1, d)
    results['acf_err_abs'] = acf_error(np.abs(dY), np.abs(dX))
    results['acf_err_sq'] = acf_error(dY ** 2, dX ** 2)

    # --- Teacher-Sigma (only if latent v provided) ---
    if heston_v is not None:
        corr, rmse = teacher_sigma_metrics(X_gen, heston_v)
        results['teacher_sigma_corr'] = corr
        results['teacher_sigma_rmse'] = rmse

    # --- TimeGAN metrics (A13, A14) ---
    # These are optional because they require TensorFlow and the original
    # TimeGAN code; wrap in try/except so they don't crash the whole suite.
    for name, func, args in [
        ('discriminative_score', compute_discriminative_score,
         (X_real, X_gen, seed)),
        ('predictive_score', compute_predictive_score,
         (X_real, X_gen, seed)),
    ]:
        try:
            results[name] = func(*args)
        except Exception as exc:
            results[name] = float('nan')
            results[f'{name}_error'] = str(exc)

    return results


# ===================================================================
# 10.  Aggregate table construction
# ===================================================================

def compute_aggregate_tables(all_results: list, output_dir: str = "results") -> None:
    """
    Build markdown and CSV tables from a list of per-seed experiment result dicts.

    Groups by (dataset, seq_len).  Each cell shows mean +/- std over seeds.
    Saves results/<dataset>_<seq_len>.md and results/<dataset>_<seq_len>.csv.
    Also produces an aggregate ranking table.
    """
    import pandas as pd
    import os
    from collections import defaultdict

    # Columns to include in tables (Tier A only — Tier B is fixed N/A)
    metric_cols = [
        'discriminative_score', 'predictive_score',
        'path_mmd2', 'terminal_mmd2', 'increment_mmd2', 'volatility_mmd',
        'terminal_swd', 'path_swd',
        'cov_error', 'mean_rmse',
        'std_error', 'kurtosis_error',
        'acf_err_abs', 'acf_err_sq',
    ]
    heston_cols = ['teacher_sigma_corr', 'teacher_sigma_rmse']

    # Tier-B placeholders
    tier_b_cols = [
        'AC Loss (N/A)', 'R Loss (N/A)', 'Traj RMSE (N/A)',
        'Ctrl RMSE (N/A)', 'Cost Gap (N/A)', 'Ctrl Corr (N/A)',
        'Route Frac (N/A)', 'Ctrl Energy (N/A)',
    ]

    df = pd.DataFrame(all_results)

    # Separate OK and FAILED
    ok = df[df['status'] == 'OK'].copy()
    failed = df[df['status'] != 'OK'][['method', 'dataset', 'seq_len', 'seed', 'error']] if len(df) > 0 else pd.DataFrame()

    groups = ok.groupby(['dataset', 'seq_len'])

    all_rankings = defaultdict(list)

    for (ds, slen), grp in groups:
        rows = []
        for method_name in sorted(grp['method'].unique()):
            mdf = grp[grp['method'] == method_name]
            row = {'Method': method_name}

            # Links
            links = {
                'timegan': ('TimeGAN', 'https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks', 'https://github.com/jsyoon0823/TimeGAN'),
                'rgan': ('RGAN/RCGAN', 'https://arxiv.org/abs/1706.02633', 'https://github.com/ratschlab/RGAN'),
                'gtgan': ('GT-GAN', 'https://arxiv.org/abs/2210.02040', 'https://github.com/Jinsung-Jeon/GT-GAN'),
                'timevae': ('TimeVAE', 'https://arxiv.org/abs/2111.08095', 'https://github.com/abudesai/timeVAE'),
                'fourierflows': ('Fourier Flows', 'https://arxiv.org/abs/2102.05644', 'https://github.com/ahmedmalaa/Fourier-flows'),
                'csdi': ('CSDI', 'https://arxiv.org/abs/2107.03575', 'https://github.com/ermongroup/CSDI'),
                'tsdiff': ('TSDiff', 'https://arxiv.org/abs/2307.11494', 'https://github.com/amazon-science/unconditional-time-series-diffusion'),
                'diffusionts': ('Diffusion-TS', 'https://arxiv.org/abs/2403.01742', 'https://github.com/Y-debug-sys/Diffusion-TS'),
            }
            name, paper, gh = links.get(method_name, (method_name, '', ''))
            row['Paper'] = f'[Paper]({paper})' if paper else ''
            row['GitHub'] = f'[GitHub]({gh})' if gh else ''

            params = mdf['num_parameters'].iloc[0] if 'num_parameters' in mdf.columns else -1
            row['# Params'] = f'{int(params):,}' if params > 0 else 'N/A'

            for col in metric_cols:
                if col in mdf.columns:
                    vals = mdf[col].dropna().values
                    row[col] = f'{np.mean(vals):.6f} ± {np.std(vals):.6f}' if len(vals) > 0 else 'FAILED'
                else:
                    row[col] = 'N/A'

            for col in heston_cols:
                if col in mdf.columns:
                    vals = mdf[col].dropna().values
                    row[col] = f'{np.mean(vals):.4f} ± {np.std(vals):.4f}' if len(vals) > 0 else 'FAILED'
                else:
                    row[col] = 'N/A'

            for col in tier_b_cols:
                row[col] = 'N/A'

            rows.append(row)

        col_order = ['Method', 'Paper', 'GitHub', '# Params'] + metric_cols
        if ds == 'heston':
            col_order += heston_cols
        col_order += tier_b_cols

        table = pd.DataFrame(rows)
        table = table[[c for c in col_order if c in table.columns]]

        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.join(output_dir, f'{ds}_{slen}')
        table.to_csv(stem + '.csv', index=False)

        with open(stem + '.md', 'w') as f:
            f.write(f'# {ds} (seq_len={slen}) — Results\n\n')
            f.write(table.to_markdown(index=False, floatfmt='.6f') + '\n')

        for _, rr in table.iterrows():
            for col in metric_cols:
                vs = rr.get(col, '')
                if '±' in vs:
                    try:
                        all_rankings[rr['Method']].append(float(vs.split(' ± ')[0]))
                    except:
                        pass

    if all_rankings:
        rank_rows = [{'Method': m, 'Avg Metric Value': np.mean(v)} for m, v in all_rankings.items()]
        rank_df = pd.DataFrame(rank_rows).sort_values('Avg Metric Value')
        rank_df.to_csv(os.path.join(output_dir, 'aggregate_ranking.csv'), index=False)
        with open(os.path.join(output_dir, 'aggregate_ranking.md'), 'w') as f:
            f.write('# Aggregate Ranking (mean Tier-A metric values)\n\n')
            f.write(rank_df.to_markdown(index=False, floatfmt='.6f') + '\n')

    if len(failed) > 0:
        failed.to_csv(os.path.join(output_dir, 'failures.csv'), index=False)
        with open(os.path.join(output_dir, 'failures.md'), 'w') as f:
            f.write(f'# Failures ({len(failed)})\n\n{failed.to_markdown(index=False)}\n')
