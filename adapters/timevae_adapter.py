"""
timeVAE Adapter for TSG Benchmark.

Wraps the timeVAE model (TensorFlow 2.x) from repos/timeVAE/ into the standard
fit / sample / num_parameters interface.

The model is a variational autoencoder with trend, seasonality, and residual
decomposition components, designed specifically for time series.
"""

import os
import sys
import time
import importlib
import numpy as np

# ── Repo import trick ──────────────────────────────────────────────────────────
_REPO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "repos", "timeVAE", "src")
)
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

# Suppress TF info/warning logs before importing TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf


# ── Lazy imports (TF is heavy) ─────────────────────────────────────────────────
def _import_timevae():
    from vae.timevae import TimeVAE
    from vae.vae_base import BaseVariationalAutoencoder
    return TimeVAE, BaseVariationalAutoencoder


# ══════════════════════════════════════════════════════════════════════════════
# Adapter
# ══════════════════════════════════════════════════════════════════════════════

class TimeVAEAdapter:
    """Adapter wrapping timeVAE's TimeVAE model for the TSG benchmark."""

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        seed: int,
        device: str,
        latent_dim: int = 8,
        reconstruction_wt: float = 3.0,
        batch_size: int = 128,
        hidden_layer_sizes=None,
        trend_poly: int = 2,
        use_residual_conn: bool = True,
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.seed = seed
        self.device = device
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.batch_size = batch_size
        self.hidden_layer_sizes = hidden_layer_sizes or [50, 100, 200]
        self.trend_poly = trend_poly
        self.use_residual_conn = use_residual_conn

        # Tracking
        self._train_time = 0.0
        self._peak_mem_mb = 0.0
        self._loss_history: dict = {"recon_loss": [], "kl_loss": []}
        self._model = None

        # The benchmark may set this to the step budget.
        self.training_steps = 5000

    # ── Public API ──────────────────────────────────────────────────────────

    def fit(self, train_data: np.ndarray) -> None:
        """Train the TimeVAE model on *train_data* (N, seq_len, n_features)."""
        t_start = time.time()

        # Seed (TF2 compat -- fallback for v1 API environments)
        try:
            tf.random.set_seed(self.seed)
        except AttributeError:
            try:
                tf.compat.v1.set_random_seed(self.seed)
            except AttributeError:
                tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

        TimeVAE, _ = _import_timevae()

        # Build model
        self._model = TimeVAE(
            seq_len=self.seq_len,
            feat_dim=self.n_features,
            latent_dim=self.latent_dim,
            reconstruction_wt=self.reconstruction_wt,
            batch_size=self.batch_size,
            hidden_layer_sizes=self.hidden_layer_sizes,
            trend_poly=self.trend_poly,
            custom_seas=None,          # no explicit seasonal decomposition
            use_residual_conn=self.use_residual_conn,
        )

        # Convert step budget to epoch budget
        n = len(train_data)
        epochs = max(1, int(self.training_steps * self.batch_size / max(n, 1)))

        # Train
        hist = self._model.fit_on_data(
            train_data,
            max_epochs=epochs,
            verbose=0,
        )
        # Capture loss history from training history
        if hist is not None and hasattr(hist, 'history'):
            for epoch, loss_val in enumerate(hist.history.get('loss', [])):
                self._loss_history["recon_loss"].append((epoch, float(loss_val)))
        # Try to extract KL loss if available
        if hist is not None and hasattr(hist, 'history'):
            for epoch, kl_val in enumerate(hist.history.get('kl_loss', [])):
                self._loss_history["kl_loss"].append((epoch, float(kl_val)))

        t_end = time.time()
        self._train_time = t_end - t_start

        # Peak GPU memory (TF)
        try:
            mem_info = tf.config.experimental.get_memory_info("GPU:0")
            self._peak_mem_mb = mem_info["peak"] / (1024 * 1024)
        except Exception:
            self._peak_mem_mb = 0.0

    def sample(self, n: int) -> np.ndarray:
        """Generate *n* synthetic samples of shape (n, seq_len, n_features)."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        samples = self._model.get_prior_samples(num_samples=n)
        # Ensure correct shape
        if samples.ndim == 2:
            samples = samples.reshape(n, self.seq_len, self.n_features)
        return samples

    def num_parameters(self) -> int:
        """Return the number of trainable parameters."""
        if self._model is None:
            return 0
        try:
            trainable, _, total = self._model.get_num_trainable_variables()
            return int(trainable)
        except (AttributeError, TypeError):
            # TF2/Keras3 compatibility: v.get_shape() -> v.shape
            return int(sum(
                np.prod(v.shape) if hasattr(v, 'shape') else 1
                for v in self._model.trainable_weights
            ))

    # ── Properties consumed by the benchmark runner ─────────────────────────

    @property
    def train_time_sec(self) -> float:
        return self._train_time

    @property
    def peak_gpu_mem_mb(self) -> float:
        return self._peak_mem_mb
