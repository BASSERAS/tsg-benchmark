"""
RGAN / RCGAN Adapter for TSG Benchmark.

Wraps the RGAN model (TF 1.x, recurrent GAN) from repos/RGAN/
into the standard fit / sample / num_parameters interface.

Uses tf.compat.v1 for TF1-style execution on TF2.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
# Ensure logger outputs to stdout
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setLevel(logging.INFO)
    logger.addHandler(_ch)

_REPO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "repos", "RGAN")
)
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow.compat.v1 as tf  # type: ignore[import-untyped]
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


# ---------------------------------------------------------------------------
# GPU memory helper
# ---------------------------------------------------------------------------

def _gpu_memory_mb() -> float:
    """Return maximum used GPU memory across visible devices (MB)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        )
        vals = [float(line.strip())
                for line in out.strip().split("\n") if line.strip()]
        return max(vals) if vals else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Scaling helpers  (data -> [-1, 1] to match tanh output, then back)
# ---------------------------------------------------------------------------

def _scale_to_unit(data: np.ndarray, min_val: np.ndarray, max_val: np.ndarray,
                   eps: float = 1e-7) -> np.ndarray:
    """Scale from [min, max] to [-1, 1]."""
    return 2.0 * (data - min_val) / (max_val - min_val + eps) - 1.0


def _inverse_scale(data: np.ndarray, min_val: np.ndarray, max_val: np.ndarray,
                   eps: float = 1e-7) -> np.ndarray:
    """Inverse of _scale_to_unit."""
    return (data + 1.0) / 2.0 * (max_val - min_val + eps) + min_val


# ===========================================================================
# Adapter class
# ===========================================================================

class RGANAdapter:
    """Adapter wrapping the RGAN model for the TSG benchmark.

    Parameters
    ----------
    seq_len : int
        Length of each time-series sequence.
    n_features : int
        Number of feature dimensions per time-step.
    seed : int
        Random seed.
    device : str
        Device string (ignored).
    latent_dim : int, optional
        Dimensionality of the latent noise space (default 8).
    hidden_units_g : int, optional
        Generator RNN hidden units (default 24).
    hidden_units_d : int, optional
        Discriminator RNN hidden units (default 24).
    batch_size : int, optional
        Training batch size (default 128).
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        seed: int,
        device: str,
        latent_dim: int = 8,
        hidden_units_g: int = 24,
        hidden_units_d: int = 24,
        batch_size: int = 128,
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.seed = seed
        self.device = device
        self.latent_dim = latent_dim
        self.hidden_units_g = hidden_units_g
        self.hidden_units_d = hidden_units_d
        self.batch_size = batch_size

        # Exposed so the benchmark runner can override this.
        self.training_steps: int = 5000

        self._train_time: float = 0.0
        self._peak_mem_mb: float = 0.0
        self._loss_history: Dict[str, list] = {"d_loss": [], "g_loss": []}
        self._graph: Optional[tf.Graph] = None
        self._sess: Optional[tf.Session] = None
        self._total_params: int = 0

        # Placeholder / output tensor references.
        self._Z_ph: Optional[tf.Tensor] = None
        self._X_ph: Optional[tf.Tensor] = None
        self._G_sample: Optional[tf.Tensor] = None

        # Normalisation stats (computed in fit).
        self._min_val: Optional[np.ndarray] = None
        self._max_val: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_data: np.ndarray) -> None:
        """Train the RGAN on *train_data*.

        Data is scaled to ``[-1, 1]`` using feature-wise min-max before
        training so it matches the generator's tanh output range.

        Parameters
        ----------
        train_data : ndarray of shape (n, seq_len, n_features)
        """
        logger.info(
            "RGAN fit starting -- %d seqs x %d steps x %d feats, "
            "h_g=%d h_d=%d z_dim=%d batch=%d epochs=%d",
            train_data.shape[0], self.seq_len, self.n_features,
            self.hidden_units_g, self.hidden_units_d,
            self.latent_dim, self.batch_size, self.training_steps,
        )

        t_start = time.perf_counter()
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        # ---- Scale data to [-1, 1] ----
        self._min_val = np.min(np.min(train_data, axis=0), axis=0)
        self._max_val = np.max(np.max(train_data, axis=0), axis=0)
        data = _scale_to_unit(train_data, self._min_val, self._max_val)

        n_obs, seq_len, n_feat = data.shape
        bsize = min(self.batch_size, n_obs)

        # ---- Build graph ----
        self._graph = tf.Graph()
        with self._graph.as_default():
            tf.set_random_seed(self.seed)

            Z = tf.placeholder(tf.float32, [None, seq_len, self.latent_dim],
                               name="Z")
            X = tf.placeholder(tf.float32, [None, seq_len, n_feat],
                               name="X")

            # --- Generator ---
            with tf.variable_scope("generator") as _:
                g_cell = tf.nn.rnn_cell.LSTMCell(
                    num_units=self.hidden_units_g, state_is_tuple=True
                )
                g_out, _ = tf.nn.dynamic_rnn(g_cell, Z, dtype=tf.float32)
                W_g = tf.get_variable(
                    "W_out_G", [self.hidden_units_g, n_feat],
                    initializer=tf.truncated_normal_initializer(),
                )
                b_g = tf.get_variable(
                    "b_out_G", [n_feat],
                    initializer=tf.truncated_normal_initializer(),
                )
                g_2d = tf.matmul(
                    tf.reshape(g_out, [-1, self.hidden_units_g]), W_g
                ) + b_g
                G_sample = tf.reshape(
                    tf.nn.tanh(g_2d), [-1, seq_len, n_feat]
                )

            # --- Discriminator ---
            def _discriminator(x: tf.Tensor,
                               reuse: bool = False
                               ) -> Tuple[tf.Tensor, tf.Tensor]:
                with tf.variable_scope("discriminator", reuse=reuse):
                    d_cell = tf.nn.rnn_cell.LSTMCell(
                        num_units=self.hidden_units_d, state_is_tuple=True
                    )
                    d_out, _ = tf.nn.dynamic_rnn(d_cell, x, dtype=tf.float32)
                    W_d = tf.get_variable(
                        "W_out_D", [self.hidden_units_d, 1],
                        initializer=tf.truncated_normal_initializer(),
                    )
                    b_d = tf.get_variable(
                        "b_out_D", [1],
                        initializer=tf.truncated_normal_initializer(),
                    )
                    logits = tf.einsum("ijk,km->ijm", d_out, W_d) + b_d
                    return tf.nn.sigmoid(logits), logits

            D_real, D_logit_real = _discriminator(X, reuse=False)
            D_fake, D_logit_fake = _discriminator(G_sample, reuse=True)

            # Losses (scalars, mean-reduced).
            D_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D_logit_real, labels=tf.ones_like(D_logit_real)
                )
            ) + tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)
                )
            )

            G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)
                )
            )

            d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope="discriminator")
            g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope="generator")

            # Solvers (Adam for both, discriminator lr 1e-4, gen lr 1e-3).
            D_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
                D_loss, var_list=d_vars
            )
            G_solver = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(
                G_loss, var_list=g_vars
            )

            init_op = tf.global_variables_initializer()
            self._total_params = sum(
                int(np.prod(v.get_shape().as_list()))
                for v in tf.trainable_variables()
            )

        # ---- Training ----
        self._sess = tf.Session(graph=self._graph)
        with self._graph.as_default():
            self._sess.run(init_op)

        n_batches = max(n_obs // bsize, 1)
        for step in range(self.training_steps):
            # 5 rounds D, 1 round G (classic GAN recipe).
            for _ in range(5):
                idx = np.random.choice(n_obs, bsize, replace=False)
                X_batch = data[idx]
                Z_batch = np.random.randn(
                    bsize, seq_len, self.latent_dim
                ).astype(np.float32)
                self._sess.run(
                    D_solver,
                    feed_dict={X: X_batch, Z: Z_batch},
                )

            idx = np.random.choice(n_obs, bsize, replace=False)
            X_batch = data[idx]
            Z_batch = np.random.randn(
                bsize, seq_len, self.latent_dim
            ).astype(np.float32)
            self._sess.run(
                G_solver,
                feed_dict={Z: Z_batch, X: X_batch},
            )

            if step % 1000 == 0 or step == self.training_steps - 1:
                d_loss_val, g_loss_val = self._sess.run(
                    [D_loss, G_loss],
                    feed_dict={X: X_batch, Z: Z_batch},
                )
                logger.info(
                    "  RGAN step %5d/%d  D_loss=%.4f  G_loss=%.4f",
                    step + 1, self.training_steps, d_loss_val, g_loss_val,
                )
                self._loss_history["d_loss"].append((step + 1, float(d_loss_val)))
                self._loss_history["g_loss"].append((step + 1, float(g_loss_val)))

        elapsed = time.perf_counter() - t_start
        self._train_time = elapsed
        self._peak_mem_mb = _gpu_memory_mb()

        # Keep tensor references for sampling.
        self._Z_ph = Z
        self._X_ph = X
        self._G_sample = G_sample

        logger.info("RGAN fit complete -- %.2f s, peak GPU %.0f MB",
                    elapsed, self._peak_mem_mb)

    # ------------------------------------------------------------------

    def sample(self, n: int) -> np.ndarray:
        """Generate *n* synthetic sequences.

        Parameters
        ----------
        n : int
            Number of sequences.

        Returns
        -------
        samples : ndarray of shape (n, seq_len, n_features)
        """
        if self._sess is None or self._G_sample is None:
            raise RuntimeError("Model not trained.  Call fit() first.")

        logger.info("RGAN sampling %d sequences ...", n)

        Z_batch = np.random.randn(
            n, self.seq_len, self.latent_dim
        ).astype(np.float32)
        samples = self._sess.run(
            self._G_sample, feed_dict={self._Z_ph: Z_batch}
        )  # values in [-1, 1]

        # Denormalise to original feature scale.
        result = _inverse_scale(samples, self._min_val, self._max_val)
        logger.info("RGAN sampled %d sequences -- shape %s", n, result.shape)
        return result.astype(np.float64)

    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        """Return total trainable parameters."""
        return int(self._total_params)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def train_time_sec(self) -> float:
        return self._train_time

    @property
    def peak_gpu_mem_mb(self) -> float:
        return self._peak_mem_mb
