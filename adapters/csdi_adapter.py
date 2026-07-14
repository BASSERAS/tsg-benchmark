"""
CSDI Adapter for TSG Benchmark.

CSDI is a conditional diffusion model for time series imputation.
For unconditional generation we set is_unconditional=True and use an
all-missing mask so the model denoises from pure noise without conditioning.
"""

import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

CSDI_REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "repos", "CSDI")
)
if CSDI_REPO not in sys.path:
    sys.path.insert(0, CSDI_REPO)


class CSDIDataset(Dataset):
    """Wraps numpy array (N, T, K) as a CSDI-compatible dataset.

    The original CSDI Forecasting_Dataset returns each item as (L, K)
    where L = seq_len (time), K = features. The DataLoader collates into
    (B, L, K), and process_data permutes to (B, K, L) for internal use.
    """

    def __init__(self, data: np.ndarray):
        self.data = torch.from_numpy(data).float()
        self.N, self.T, self.K = data.shape

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.data[idx]  # (T, K) — same as original (L, K)
        # Random masking for unconditional training:
        # gt_mask=1 means observed (used for conditioning),
        # gt_mask=0 means missing (loss is computed on those positions).
        # Each sample gets a different random missing ratio so the model
        # learns the unconditional score function across all positions.
        missing_ratio = np.random.uniform(0.2, 0.8)
        gt_mask = (torch.rand(self.T, self.K) >= missing_ratio).float()
        return {
            "observed_data": x,  # (T, K) — process_data permutes to (K, T)
            "observed_mask": torch.ones(self.T, self.K),  # (T, K)
            "gt_mask": gt_mask,  # (T, K), random missing pattern
            "timepoints": torch.arange(self.T).int(),  # (T,)
            "observed_tp": torch.arange(self.T).float(),  # (T,)
        }


class CSDIAdapter:
    """Adapter wrapping CSDI for the TSG benchmark."""

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        seed: int,
        device: str,
        batch_size: int = 128,
        lr: float = 1e-3,
        num_steps: int = 50,
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.seed = seed
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.lr = lr
        self.num_steps = num_steps
        self._train_time = 0.0
        self._peak_mem_mb = 0.0
        self._loss_history: dict = {"diffusion_loss": []}
        self._model = None
        self.training_steps = 5000

    def fit(self, train_data: np.ndarray) -> None:
        t_start = time.time()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        from main_model import CSDI_Forecasting

        config = {
            "model": {
                "timeemb": 128,
                "featureemb": 16,
                "is_unconditional": True, "num_sample_features": 64,
                "target_strategy": "random",
            },
            "diffusion": {
                "num_steps": self.num_steps,
                "schedule": "quad",
                "beta_start": 0.0001,
                "beta_end": 0.5,
                "layers": 4, "channels": 64, "nheads": 8, "diffusion_embedding_dim": 128, "is_linear": True,
            },
            "train": {
                "batch_size": self.batch_size,
                "epochs": max(1, self.training_steps * self.batch_size // len(train_data)),
                "lr": self.lr,
            },
        }

        self._model = CSDI_Forecasting(config, self.device, self.n_features).to(self.device)

        dataset = CSDIDataset(train_data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        self._model.train()

        step = 0
        for epoch in range(config["train"]["epochs"]):
            loss = None
            for batch in loader:
                if step >= self.training_steps:
                    break
                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                loss = self._model(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

            if epoch % 5 == 0 and loss is not None:
                print(f"  CSDI epoch {epoch}/{config['train']['epochs']}, step {step}, loss: {loss.item():.4f}")
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
                # All-missing mask: cond_mask = zeros
                cond_mask = torch.zeros(bs, self.n_features, self.seq_len, device=self.device)
                observed_tp = torch.arange(self.seq_len, device=self.device).float().unsqueeze(0).expand(bs, -1)

                # Build side_info tensor using the model's method (not a dict)
                side_info = self._model.get_side_info(observed_tp, cond_mask)

                # Start from pure noise (B, K, L) and impute with n_samples=1
                noise = torch.randn(bs, self.n_features, self.seq_len, device=self.device)
                gen = self._model.impute(noise, cond_mask, side_info, 1)
                gen = gen[:, 0]  # (B, K, L), collapse n_samples dim
                gen = gen.permute(0, 2, 1)  # (B, L, K) = (B, T, K)
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
