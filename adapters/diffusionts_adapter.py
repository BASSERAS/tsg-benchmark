"""
Diffusion-TS Adapter for TSG Benchmark.

Wraps the Diffusion-TS model (interpretable diffusion with Transformer backbone)
from repos/Diffusion-TS/ into the standard fit / sample / num_parameters interface.
"""

import os
import sys
import time
import warnings
import math

import numpy as np

_REPO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "repos", "Diffusion-TS")
)
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DiffusionTSAdapter:
    """Adapter wrapping Diffusion-TS for the TSG benchmark."""

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        seed: int,
        device: str,
        batch_size: int = 128,
        lr: float = 1e-3,
        timesteps: int = 50,
        d_model: int = 64,
        n_layer: int = 4,
        n_heads: int = 4,
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.seed = seed
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.lr = lr
        self.timesteps = timesteps
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_heads = n_heads

        self._train_time = 0.0
        self._peak_mem_mb = 0.0
        self._loss_history: dict = {"diffusion_loss": []}
        self._model = None
        self.training_steps = 5000

    def _build_model(self):
        from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
        self._model = Diffusion_TS(
            seq_length=self.seq_len,
            feature_size=self.n_features,
            timesteps=self.timesteps,
            sampling_timesteps=self.timesteps,
            loss_type='l1',
            beta_schedule='cosine',
            n_layer_enc=self.n_layer,
            n_layer_dec=self.n_layer,
            d_model=self.d_model,
            n_heads=self.n_heads,
            mlp_hidden_times=2,
        ).to(self.device)

    def fit(self, train_data: np.ndarray) -> None:
        t_start = time.time()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._build_model()

        dataset = TensorDataset(torch.from_numpy(train_data).float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr)

        self._model.train()
        step = 0
        while step < self.training_steps:
            for batch in loader:
                if step >= self.training_steps:
                    break
                x = batch[0].to(self.device)
                loss = self._model(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                if step % 1000 == 0:
                    print(f"  Diffusion-TS step {step}/{self.training_steps}, loss: {loss.item():.6f}")
                    self._loss_history["diffusion_loss"].append((step, loss.item()))

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
                shape = (bs, self.seq_len, self.n_features)
                gen = self._model.sample(shape)
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
