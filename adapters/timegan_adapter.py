"""
TimeGAN Adapter for TSG Benchmark.

Wraps the TimeGAN model (TF 1.15) from repos/TimeGAN/ into the standard
fit / sample / num_parameters interface.  Uses TF 1.x directly (the
runner activates a ``tf1_env`` conda environment).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_REPO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "repos", "TimeGAN")
)
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf  # TF 1.x
from utils import batch_generator, random_generator  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# RNN cell factory (avoids ``tf.contrib`` which may be absent)
# ---------------------------------------------------------------------------

def _rnn_cell(module_name: str, hidden_dim: int) -> tf.nn.rnn_cell.RNNCell:
    """Create a single RNN cell of the requested type."""
    if module_name == "gru":
        return tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
    elif module_name == "lstm":
        return tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim)
    else:
        raise ValueError(f"Unsupported RNN module: {module_name!r}")


# ---------------------------------------------------------------------------
# Scaler helpers  (feature-wise min-max, matching TimeGAN's internal logic)
# ---------------------------------------------------------------------------

def _compute_min_max(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Feature-wise min / max across samples and timesteps."""
    return (
        np.min(np.min(data, axis=0), axis=0),
        np.max(np.max(data, axis=0), axis=0),
    )


def _normalise(data: np.ndarray, min_val: np.ndarray, max_val: np.ndarray,
               eps: float = 1e-7) -> np.ndarray:
    return (data - min_val) / (max_val - min_val + eps)


def _renormalise(data: np.ndarray, min_val: np.ndarray, max_val: np.ndarray,
                 eps: float = 1e-7) -> np.ndarray:
    return data * (max_val - min_val + eps) + min_val


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


# ===========================================================================
# Adapter class
# ===========================================================================

class TimeGANAdapter:
    """Adapter wrapping TimeGAN for the TSG benchmark.

    Parameters
    ----------
    seq_len : int
        Length of each time-series sequence.
    n_features : int
        Number of feature dimensions per time-step.
    seed : int
        Random seed.
    device : str
        Device string (ignored — GPU selection via ``CUDA_VISIBLE_DEVICES``).
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        seed: int,
        device: str,
        hidden_dim: int = 24,
        num_layer: int = 3,
        batch_size: int = 128,
        module: str = "gru",
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.seed = seed
        self.device = device

        # Exposed so the benchmark runner can override the step budget.
        self.training_steps: int = 5000

        self._params: Dict[str, Any] = dict(
            module=module,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            iterations=self.training_steps,
            batch_size=batch_size,
        )
        self._train_time: float = 0.0
        self._peak_mem: float = 0.0

        # ---- State populated by fit() ----
        self._graph: Optional[tf.Graph] = None
        self._sess: Optional[tf.Session] = None

        # Normalisation stats captured during fit().
        self._min_val: Optional[np.ndarray] = None
        self._max_val: Optional[np.ndarray] = None

        # Tensor references recovered from the trained graph.
        self._X_ph: Optional[tf.Tensor] = None
        self._Z_ph: Optional[tf.Tensor] = None
        self._T_ph: Optional[tf.Tensor] = None
        self._X_hat: Optional[tf.Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_data: np.ndarray) -> None:
        """Fit TimeGAN to *train_data*.

        Data is normalised internally via feature-wise MinMax scaling.
        The generated samples returned by ``timegan()`` are discarded —
        only the trained graph and session are retained.

        Parameters
        ----------
        train_data : ndarray of shape (n_sequences, seq_len, n_features)
            Time-series data.  May be raw or pre-scaled.
        """
        t_start = time.perf_counter()
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        # Sync parameters with external step budget override.
        self._params["iterations"] = self.training_steps

        n_obs = train_data.shape[0]
        self._params["batch_size"] = min(
            self._params["batch_size"], n_obs
        )

        logger.info(
            "TimeGAN fit starting -- %d seqs x %d steps x %d feats, "
            "hidden=%d layers=%d iters=%d batch=%d",
            n_obs, self.seq_len, self.n_features,
            self._params["hidden_dim"], self._params["num_layer"],
            self._params["iterations"], self._params["batch_size"],
        )

        # Convert to list-of-arrays as expected by TimeGAN internals.
        ori_data: list = [train_data[i] for i in range(n_obs)]

        # Close any stale session from a previous fit().
        if self._sess is not None:
            self._sess.close()
            self._sess = None

        # ---- Monkey-patch tf.Session -> tf.InteractiveSession so the   ----
        # ---- session created inside timegan() registers as the default  ----
        # ---- and stays accessible after the function returns.           ----
        _orig_Session = tf.Session
        tf.Session = tf.InteractiveSession  # type: ignore[assignment]

        try:
            from timegan import timegan  # type: ignore[import-untyped]
            _generated = timegan(ori_data, self._params)
        finally:
            tf.Session = _orig_Session

        elapsed = time.perf_counter() - t_start
        self._train_time = elapsed
        self._graph = tf.get_default_graph()
        self._sess = tf.get_default_session()

        # Capture normalisation stats (same formula as timegan's MinMaxScaler).
        data_arr = np.asarray(ori_data)
        self._min_val = np.min(np.min(data_arr, axis=0), axis=0)
        self._max_val = np.max(np.max(data_arr, axis=0), axis=0)

        # Locate placeholders and the X_hat tensor in the trained graph.
        self._locate_tensors()

        # GPU memory -- query after training.
        self._peak_mem = _gpu_memory_mb()

        logger.info(
            "TimeGAN fit complete -- %.2f s, peak GPU %.0f MB",
            elapsed, self._peak_mem,
        )

    # ------------------------------------------------------------------

    def sample(self, n: int) -> np.ndarray:
        """Generate *n* synthetic sequences.

        Parameters
        ----------
        n : int
            Number of sequences to generate.

        Returns
        -------
        samples : ndarray of shape (n, seq_len, n_features)
        """
        if self._sess is None or self._graph is None:
            raise RuntimeError("No trained model.  Call fit() first.")
        if self._X_hat is None:
            raise RuntimeError(
                "Could not locate the X_hat tensor in the trained graph."
            )

        logger.info("TimeGAN sampling %d sequences ...", n)

        # Build random noise.  In TimeGAN ``z_dim == dim`` (the feature count).
        ori_time = [self.seq_len] * n
        Z_mb = random_generator(n, self.n_features, ori_time, self.seq_len)

        with self._graph.as_default():
            gen = self._sess.run(
                self._X_hat,
                feed_dict={
                    self._Z_ph: Z_mb,
                    self._T_ph: ori_time,
                    self._X_ph: np.zeros(
                        (n, self.seq_len, self.n_features), dtype=np.float32
                    ),
                },
            )

        # gen may be a list of variable-length arrays padded to max_seq_len.
        # Stack and trim to the expected shape.
        if isinstance(gen, list):
            out = np.zeros((n, self.seq_len, self.n_features),
                           dtype=np.float64)
            for i, arr in enumerate(gen):
                L = min(arr.shape[0], self.seq_len)
                out[i, :L] = arr[:L]
        else:
            out = np.asarray(gen, dtype=np.float64)
            if out.ndim == 3 and out.shape[1] != self.seq_len:
                L = min(out.shape[1], self.seq_len)
                out = out[:, :L, :]

        # Denormalise to original scale (timegan() returns normalised data).
        out = _renormalise(out, self._min_val, self._max_val)

        logger.info("TimeGAN sampled %d sequences -- shape %s", n, out.shape)
        return out.astype(np.float64)

    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        if self._graph is None:
            # Fallback estimate.
            h = self._params["hidden_dim"]
            nl = self._params["num_layer"]
            return int((h * h * 4 * nl) * 4)

        with self._graph.as_default():
            total = 0
            for var in tf.trainable_variables():
                v = 1
                for d in var.get_shape().as_list():
                    v *= d
                total += v
            return total

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def train_time_sec(self) -> float:
        return self._train_time

    @property
    def peak_gpu_mem_mb(self) -> float:
        return self._peak_mem

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _locate_tensors(self) -> None:
        """Find the X, Z, T placeholders and X_hat output in the graph.

        The graph built by ``timegan()`` uses specifically named
        placeholders:

          * ``myinput_x:0``   -- data placeholder ``X``
          * ``myinput_z:0``   -- noise placeholder ``Z``
          * ``myinput_t:0``   -- sequence-length placeholder ``T``

        ``X_hat`` is the output of the second ``recovery`` RNN (the one
        that takes ``H_hat`` as input).  It is
        ``recovery/fully_connected_1/Sigmoid:0``
        (the second ``tf.layers.dense`` under the ``recovery`` scope).
        """
        if self._graph is None:
            return

        with self._graph.as_default():
            try:
                self._X_ph = self._graph.get_tensor_by_name("myinput_x:0")
                self._Z_ph = self._graph.get_tensor_by_name("myinput_z:0")
                self._T_ph = self._graph.get_tensor_by_name("myinput_t:0")
            except KeyError as exc:
                logger.warning("Could not find placeholders: %s", exc)
                return

            # Find X_hat: the second Sigmoid under ``recovery/``.
            xhat_candidates = []
            for op in self._graph.get_operations():
                if (op.type == "Sigmoid"
                        and "recovery" in op.name
                        and "fully_connected" in op.name):
                    xhat_candidates.append(op.outputs[0])

            if len(xhat_candidates) >= 2:
                # ``recovery/fully_connected/Sigmoid``  -- X_tilde
                # ``recovery/fully_connected_1/Sigmoid`` -- X_hat
                self._X_hat = xhat_candidates[-1]
            elif len(xhat_candidates) == 1:
                self._X_hat = xhat_candidates[0]
            else:
                # Fall back to any recovery sigmoid.
                for op in self._graph.get_operations():
                    if op.type == "Sigmoid" and "recovery" in op.name:
                        self._X_hat = op.outputs[0]
                        break
