"""
Fourier-flows Adapter for TSG Benchmark.

Wraps the FourierFlow model (PyTorch, normalizing flow with spectral filters)
from repos/Fourier-flows/SequentialFlows.py into the standard fit / sample /
num_parameters interface.

IMPORTANT DESIGN NOTES
----------------------
The DFT module in FourierFlow uses numpy FFT internally (np.fft.fft,
np.fft.ifft), so the model MUST stay on CPU regardless of the `device` arg.
The adapter prints a warning when device != "cpu".

Data shape handling:
  The FourierFlow model expects single-channel time series (batch, seq_len).
  For multi-feature data (batch, seq_len, n_features), we reshape to
  (batch * n_features, seq_len) and train a single flow on all feature
  channels independently.  During sampling we generate per-feature
  sequences and reshape back.

  fft_size is set to _next_odd(seq_len) so the DFT / spectral filter
  splitting works correctly with the .view(-1, d+1) reshape.

  An epsilon (1e-8) is added to fft_std during spectral normalisation
  to prevent division by zero: the imaginary component of the DC bin
  is always 0 for real-valued signals.
"""

import os
import sys
import time
import warnings

import numpy as np

# ── Repo import trick ──────────────────────────────────────────────────────────
_REPO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "repos", "Fourier-flows")
)
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

import torch


def _import_fourierflow():
    from SequentialFlows import FourierFlow
    return FourierFlow


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _next_odd(n: int) -> int:
    """Return the smallest odd integer >= n."""
    return n if (n % 2 == 1) else n + 1


def _pad_to(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pad last dimension of *x* from current dim to *target_dim* by
    repeating the last value along that dimension."""
    cur = x.shape[-1]
    if cur >= target_dim:
        return x
    n_pad = target_dim - cur
    last_col = x[..., -1:]                     # (..., 1)
    pad = last_col.expand(*x.shape[:-1], n_pad)  # (..., n_pad)
    return torch.cat([x, pad], dim=-1)


def _crop_to(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Crop the last dimension to *target_dim*."""
    if x.shape[-1] <= target_dim:
        return x
    return x[..., :target_dim]


# ══════════════════════════════════════════════════════════════════════════════
# Adapter
# ══════════════════════════════════════════════════════════════════════════════

class FourierFlowsAdapter:
    """Adapter wrapping FourierFlow for the TSG benchmark.

    Parameters
    ----------
    seq_len : int
        Length of the time series.
    n_features : int
        Number of feature channels per time step.
    seed : int
        Random seed.
    device : str
        Device string (ignored — FourierFlow is CPU-only due to numpy FFT).
    hidden : int
        Number of hidden units in each spectral filter (default 200).
    n_flows : int
        Number of stacked spectral-filter bijectors (default 3).
    normalize : bool
        Whether to z-normalise spectral features (default True).
    flip : bool
        Whether to alternate the split between even/odd flows (default True).
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        seed: int,
        device: str,
        hidden: int = 200,
        n_flows: int = 3,
        normalize: bool = True,
        flip: bool = True,
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.seed = seed
        self.hidden = hidden
        self.n_flows = n_flows
        self.normalize = normalize
        self.flip = flip

        # Warn about CPU-only
        if device.lower() != "cpu":
            warnings.warn(
                "FourierFlow uses numpy-based DFT internally and must run on "
                f"CPU. Ignoring device='{device}'.",
                RuntimeWarning,
            )
        self._device = "cpu"

        # FFT size equals next odd >= seq_len.  The flow operates on
        # single-channel time series (batch, seq_len).  Multi-feature data
        # is reshaped to (batch * n_features, seq_len) for training.
        self._fft_size = _next_odd(seq_len)

        # Tracking
        self._train_time = 0.0
        self._peak_mem_mb = 0.0
        self._loss_history: dict = {"nll_loss": []}
        self._model = None

        # The benchmark may set this to the step budget.
        self.training_steps = 5000

    # ── Public API ──────────────────────────────────────────────────────────

    def fit(self, train_data: np.ndarray) -> None:
        """Train FourierFlow on *train_data* (N, seq_len, n_features).

        Reshapes multi-feature data to (N * n_features, seq_len) so the flow
        treats each feature channel as an independent time series, pads to
        fft_size (next odd >= seq_len), then trains with the custom loop.
        """
        t_start = time.time()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        FourierFlow = _import_fourierflow()

        # Reshape multi-feature data: (N, seq_len, n_features) -> (N*n_features, seq_len)
        n_samples, t_len, n_feat = train_data.shape
        flat_data = train_data.transpose(0, 2, 1).reshape(-1, t_len)  # (N*D, T)

        # Pad to fft_size (N*n_features, fft_size)
        padded = np.zeros((flat_data.shape[0], self._fft_size), dtype=np.float32)
        dim = min(flat_data.shape[-1], self._fft_size)
        padded[:, :dim] = flat_data[:, :dim]
        if flat_data.shape[-1] < self._fft_size:
            padded[:, flat_data.shape[-1]:] = flat_data[:, -1:]

        # Build model
        self._model = FourierFlow(
            hidden=self.hidden,
            fft_size=self._fft_size,
            n_flows=self.n_flows,
            FFT=True,
            flip=self.flip,
            normalize=self.normalize,
        )

        # Compute spectral normalisation stats (+ epsilon to prevent division
        # by zero when a spectral bin (e.g. Nyquist imag) has zero variance).
        X_train_t = torch.from_numpy(padded).float()
        X_train_spectral = self._model.FourierTransform(X_train_t)[0]
        self._model.fft_mean = torch.mean(X_train_spectral, dim=0)
        self._model.fft_std = torch.std(X_train_spectral, dim=0) + 1e-8

        # Optimiser
        epochs = self.training_steps
        optimiser = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, 0.999)

        batch_size = min(128, padded.shape[0])

        for step in range(1, epochs + 1):
            # Mini-batch
            idx = torch.randperm(padded.shape[0])[:batch_size]
            X_batch = torch.from_numpy(padded[idx]).float()

            optimiser.zero_grad()

            z, log_pz, log_jacob = self._model(X_batch)
            loss = (-log_pz - log_jacob).mean()

            loss.backward()
            # Gradient clipping to prevent numerical overflow from
            # exp(sig) in the spectral coupling layers (the sigmoid
            # network's final linear layer can produce arbitrarily
            # large values that overflow when exponentiated).
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 10.0)
            optimiser.step()
            scheduler.step()

            if step % 500 == 0 or step == epochs:
                print(
                    f"  [FourierFlows] Step {step}/{epochs}, "
                    f"loss: {loss.item():.4f}"
                )
                self._loss_history["nll_loss"].append((step, loss.item()))

        print("  [FourierFlows] Training done.")

        t_end = time.time()
        self._train_time = t_end - t_start
        self._peak_mem_mb = 0.0  # CPU-only

    def sample(self, n: int) -> np.ndarray:
        """Generate *n* synthetic samples (n, seq_len, n_features).

        Generates n * n_features single-channel sequences from the flow,
        crops each to seq_len, then reshapes to (n, seq_len, n_features).
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        self._model.eval()

        # FourierFlow.sample() draws from N(0, I), applies inverse spectral
        # filters and inverse DFT, returning (n_samples, fft_size).
        # We need n * n_features sequences to fill the (n, seq_len, n_features) shape.
        with torch.no_grad():
            raw = self._model.sample(n * self.n_features)  # (N*D, fft_size)

        # Crop from fft_size -> seq_len
        per_feat = raw[:, : self.seq_len]  # (N*D, seq_len)

        # Reshape: (N*D, seq_len) -> (n, n_features, seq_len) -> (n, seq_len, n_features)
        return per_feat.reshape(n, self.n_features, self.seq_len).transpose(0, 2, 1)

    def num_parameters(self) -> int:
        """Return the number of trainable parameters."""
        if self._model is None:
            return 0
        return sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def train_time_sec(self) -> float:
        return self._train_time

    @property
    def peak_gpu_mem_mb(self) -> float:
        return self._peak_mem_mb
