"""
TSDiff Adapter for TSG Benchmark.

Wraps the TSDiff model (PyTorch Lightning, unconditional diffusion with S4 backbone)
from repos/tsdiff/ into the standard fit / sample / num_parameters interface.
"""

import os
import sys
import time
import warnings

import numpy as np

_REPO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "repos", "tsdiff", "src")
)
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

# Patch gluonts module path changes (gluonts>=0.15 moved scaler modules)
from . import tsdiff_patch  # noqa: F401

import torch
from torch.utils.data import DataLoader, TensorDataset

# TSDiff uses GluonTS lazily; we avoid importing it at module level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRBAC_BACKEND'] = 'cpu'


class TSDiffAdapter:
    """Adapter wrapping TSDiff for the TSG benchmark."""

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        seed: int,
        device: str,
        batch_size: int = 128,
        lr: float = 1e-3,
        timesteps: int = 100,
        hidden_dim: int = 64,
        num_blocks: int = 3,
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.seed = seed
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.lr = lr
        self.timesteps = timesteps
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self._train_time = 0.0
        self._peak_mem_mb = 0.0
        self._model = None
        self.training_steps = 5000

    def _build_model(self):
        from uncond_ts_diff.model import TSDiff
        from uncond_ts_diff.utils import linear_beta_schedule

        self._model = TSDiff(
            backbone_parameters={
                "input_dim": self.n_features,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.n_features,
                "step_emb": 128,
                "num_residual_blocks": self.num_blocks,
                "residual_block": "s4",
            },
            timesteps=self.timesteps,
            diffusion_scheduler=linear_beta_schedule,
            context_length=0,
            prediction_length=self.seq_len,
            freq="H",
            normalization="none",
            use_features=False,
            use_lags=False,
            lr=self.lr,
        )

    def _p_sample_loop(self, noise: torch.Tensor) -> torch.Tensor:
        """Denoise from pure noise to a clean sample. Replaces NaN with 0."""
        batch_size = noise.shape[0]
        seq = noise
        for i in reversed(range(self._model.timesteps)):
            t = torch.full((batch_size,), i, device=noise.device, dtype=torch.long)
            seq = self._model.p_sample(seq, t, i, features=None)
            if torch.isnan(seq).any():
                seq = torch.nan_to_num(seq, nan=0.0, posinf=10.0, neginf=-10.0)
        return seq

    def fit(self, train_data: np.ndarray) -> None:
        t_start = time.time()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._build_model()
        self._model.to(self.device)

        # Simple training loop: treat data as (batch, seq_len, n_features)
        dataset = TensorDataset(torch.from_numpy(train_data).float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        self._model.train()
        step = 0
        while step < self.training_steps:
            for batch in loader:
                if step >= self.training_steps:
                    break
                x = batch[0].to(self.device)  # (batch, seq_len, n_features)

                # Direct diffusion loss: bypass GluonTS dict-based training_step.
                # p_losses accepts raw (B, L, C) tensors with features=None.
                t = torch.randint(
                    0, self._model.timesteps, (x.shape[0],), device=self.device
                ).long()
                loss, _, _ = self._model.p_losses(x, t, features=None, loss_type="l2")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                if step % 1000 == 0:
                    print(f"  TSDiff step {step}/{self.training_steps}, loss: {loss.item():.6f}")

        self._train_time = time.time() - t_start
        self._model.eval()

    def sample(self, n: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained.")
        self._model.to(self.device)
        samples = []

        with torch.no_grad():
            remaining = n
            while remaining > 0:
                bs = min(self.batch_size, remaining)
                noise = torch.randn(bs, self.seq_len, self.n_features, device=self.device)
                gen = self._p_sample_loop(noise)
                samples.append(gen.cpu().numpy())
                remaining -= bs

        result = np.concatenate(samples, axis=0)
        return np.clip(result, 0.0, 1.0).astype(np.float64)

    def num_parameters(self) -> int:
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    @property
    def train_time_sec(self) -> float:
        return self._train_time

    @property
    def peak_gpu_mem_mb(self) -> float:
        return self._peak_mem_mb
