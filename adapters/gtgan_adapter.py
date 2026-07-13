"""
GT-GAN Adapter for TSG Benchmark.

Wraps the GT-GAN model (PyTorch, vendored controldiffeq + Neural CDE + CNF
generator) from repos/GT-GAN/ into the standard fit / sample / num_parameters
interface.

Architecture (two-phase training):
  1. Embedding network training  -- NeuralCDE embedder + ODE-Net recovery
  2. Joint adversarial training  -- add CNF generator + ODE-Net discriminator
"""

import os
import sys
import time
import math
import pathlib
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

# ── Repo import trick ──────────────────────────────────────────────────────────
_GTGAN_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "repos", "GT-GAN")
)
if _GTGAN_DIR not in sys.path:
    sys.path.insert(0, _GTGAN_DIR)


# Lazy imports
def _import_gtgan_modules():
    import controldiffeq
    from ctfp_tools import (
        run_latent_ctfp_model5_3 as run_model,
        parse_arguments,
    )
    from train_misc import build_model_tabular_nonlinear
    from train_misc import (
        set_cnf_options,
        create_regularization_fns,
        count_parameters as count_cnf_params,
    )
    return (
        controldiffeq,
        run_model,
        parse_arguments,
        build_model_tabular_nonlinear,
        set_cnf_options,
        create_regularization_fns,
        count_cnf_params,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Model components (adapted from GTGAN_stocks.py)
# ══════════════════════════════════════════════════════════════════════════════

class FinalTanh(nn.Module):
    """Helper for the Neural CDE vector field."""
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels,
                 num_hidden_layers):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = nn.ModuleList(
            nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
            for _ in range(num_hidden_layers - 1))
        self.linear_out = nn.Linear(
            hidden_hidden_channels, input_channels * hidden_channels)

    def forward(self, z):
        z = self.linear_in(z).relu()
        for linear in self.linears:
            z = linear(z).relu()
        z = self.linear_out(z).view(
            *z.shape[:-1], self.hidden_channels, self.input_channels)
        return z.tanh()


class NeuralCDE(nn.Module):
    """Neural CDE: wrapper around controldiffeq.cdeint."""
    def __init__(self, func, input_channels, hidden_channels, output_channels,
                 initial=True):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.func = func
        self.initial = initial
        if initial:
            self.initial_network = nn.Linear(input_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, output_channels)
        self.activation_fn = torch.sigmoid

    def forward(self, times, coeffs, final_index, **kwargs):
        import controldiffeq
        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs)
        batch_dims = coeffs[0].shape[:-2]
        z0 = self.initial_network(cubic_spline.evaluate(times[0]))
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            kwargs.setdefault('options', {})
            if 'step_size' not in kwargs['options'] and 'grid_constructor' not in kwargs['options']:
                time_diffs = times[1:] - times[:-1]
                kwargs['options']['step_size'] = time_diffs.min().item()
        z_t = controldiffeq.cdeint(
            dX_dt=cubic_spline.derivative, z0=z0, func=self.func,
            t=times, **kwargs)
        # reshape to (..., times, channels)
        for i in range(len(z_t.shape) - 2, 0, -1):
            z_t = z_t.transpose(0, i)
        pred_y = self.linear(z_t)
        pred_y = self.activation_fn(pred_y)
        return pred_y


class _GRUODECell(nn.Module):
    """Simplified GRU-ODE cell from the vendored gru_ode_bayes."""
    def __init__(self, hidden_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.bias = bias
        self.lin_hh = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, t, h):
        return torch.tanh(self.lin_hh(h))


class _ODENetworkLayer(nn.Module):
    """One layer in the Multi_Layer_ODENetwork."""
    def __init__(self, input_size, hidden_size, output_size, delta_t,
                 last_activation='identity', is_first=False, is_last=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.delta_t = delta_t
        self.is_first = is_first
        self.is_last = is_last

        if is_first:
            self.x_model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            self.x_model = nn.Identity()

        self.gru_layer = _GRUODECell(hidden_size)
        # Use GRU instead of GRUODE's GRU for the observation update
        self.gru_obs = nn.GRU(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

        self._has_linear = is_last
        if is_last:
            if last_activation == 'identity':
                self.last_activation_fn = nn.Identity()
            elif last_activation == 'tanh':
                self.last_activation_fn = nn.Tanh()
            elif last_activation == 'sigmoid':
                self.last_activation_fn = nn.Sigmoid()
            else:
                self.last_activation_fn = nn.Identity()
            self.rec_linear = nn.Linear(hidden_size, output_size)

    def forward(self, H, times):
        HH = self.x_model(H)
        out = torch.zeros_like(HH if not self.is_first else HH)
        h = torch.zeros(HH.shape[0], self.hidden_size, device=HH.device)
        for idx in range(times.shape[1]):
            current_out, h_updated = self.gru_obs(
                HH[:, idx:idx+1, :], h.unsqueeze(0))
            h = h_updated.squeeze(0)
            out[:, idx, :] = out[:, idx, :] + current_out.squeeze(1)
        if self._has_linear:
            X_tilde = self.rec_linear(out)
            X_tilde = self.last_activation_fn(X_tilde)
            return X_tilde
        return out


class _MultiLayerODENet(nn.Module):
    """Multi-layer ODE network (recovery / discriminator)."""
    def __init__(self, input_size, hidden_size, output_size, num_layer,
                 delta_t=0.5, last_activation='identity'):
        super().__init__()
        self.num_layer = num_layer
        if num_layer == 1:
            self.model = _ODENetworkLayer(
                input_size, hidden_size, output_size, delta_t,
                last_activation=last_activation, is_first=True, is_last=True)
        else:
            layers = []
            for i in range(num_layer):
                is_first = (i == 0)
                is_last = (i == num_layer - 1)
                ins = input_size if is_first else hidden_size
                outs = output_size if is_last else hidden_size
                act = last_activation if is_last else 'identity'
                layers.append(_ODENetworkLayer(
                    ins, hidden_size, outs, delta_t,
                    last_activation=act, is_first=is_first, is_last=is_last))
            self.model = nn.ModuleList(layers)

    def forward(self, H, times):
        if self.num_layer == 1:
            return self.model(H, times)
        out = H
        for layer in self.model:
            out = layer(out, times)
        return out


class _Net(nn.Module):
    """Simple GRU-based network (supervisor / auxiliary)."""
    def __init__(self, input_size, hidden_size, output_size, num_layers,
                 activation_fn=torch.sigmoid):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation_fn = activation_fn

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Adapter
# ══════════════════════════════════════════════════════════════════════════════

class GTGANAdapter:
    """Adapter wrapping GT-GAN for the TSG benchmark."""

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        seed: int,
        device: str,
        hidden_dim: int = 24,
        latent_dim: int = 24,
        num_rec_layers: int = 2,
        num_disc_layers: int = 1,
        batch_size: int = 128,
        first_epochs: int = 10000,
        learning_rate: float = 1e-3,
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.seed = seed
        self.device = device
        self.hidden_dim = min(hidden_dim, max(8, n_features * 2))
        self.latent_dim = max(self.hidden_dim, 8)
        self.num_rec_layers = num_rec_layers
        self.num_disc_layers = num_disc_layers
        self.batch_size = batch_size
        self._first_epochs = first_epochs
        self._lr = learning_rate

        # Tracking
        self._train_time = 0.0
        self._peak_mem_mb = 0.0
        self._models = {}

        # The benchmark may set this to the step budget.
        self.training_steps = 5000

    # ── Public API ──────────────────────────────────────────────────────────

    def fit(self, train_data: np.ndarray) -> None:
        """Train GT-GAN on *train_data* (N, seq_len, n_features).

        Two-phase training:
          Phase 1 — Embedding network (autoencoder): embedder + recovery
          Phase 2 — Joint adversarial training: add generator + discriminator
        """
        t_start = time.time()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        n_data = len(train_data)

        # Determine step budget
        max_steps = self.training_steps

        # ── Build models ────────────────────────────────────────────────────
        models = self._build_models(device)
        embedder = models["embedder"]
        recovery = models["recovery"]
        generator = models["generator"]
        supervisor = models["supervisor"]
        discriminator = models["discriminator"]

        # ── Create time tensor ──────────────────────────────────────────────
        time_vals = torch.arange(self.seq_len, dtype=torch.float32, device=device)
        final_index = (torch.ones(self.batch_size, dtype=torch.long, device=device)
                       * (self.seq_len - 1))

        # ── Phase 1: Embedding Network Training ─────────────────────────────
        optimizer_er = optim.Adam(
            chain(embedder.parameters(), recovery.parameters()),
            lr=self._lr,
        )

        embedder.train()
        recovery.train()

        first_epochs = min(self._first_epochs, max_steps // 2)

        for step in range(1, first_epochs + 1):
            x_batch = self._get_batch(train_data, self.batch_size, device)
            # x_batch: (batch, seq_len, n_features)
            coeffs = self._make_coeffs(time_vals, x_batch)

            h = embedder(time_vals, coeffs, final_index)
            x_tilde = recovery(h, time_vals.unsqueeze(0).unsqueeze(2).expand(
                self.batch_size, -1, 1))

            loss_e = F.mse_loss(x_tilde, x_batch)

            optimizer_er.zero_grad()
            loss_e.backward()
            optimizer_er.step()

            if step % 500 == 0:
                print(f"  [GT-GAN] Phase 1 step {step}/{first_epochs}, "
                      f"loss_e: {math.sqrt(loss_e.item()):.4f}")

        print("  [GT-GAN] Phase 1 done.")

        # ── Phase 2: Joint Training ─────────────────────────────────────────
        optimizer_gs = optim.Adam(generator.parameters(), lr=self._lr)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=self._lr)

        embedder.eval()
        recovery.eval()
        generator.train()
        supervisor.train()
        recovery.train()
        discriminator.train()

        n_joint = max_steps - first_epochs

        for step in range(1, n_joint + 1):
            # --- Discriminator update ---
            for _ in range(2):
                x_batch = self._get_batch(train_data, self.batch_size, device)
                coeffs = self._make_coeffs(time_vals, x_batch)
                z = torch.randn(self.batch_size, self.seq_len, self.latent_dim,
                                device=device)

                with torch.no_grad():
                    h = embedder(time_vals, coeffs, final_index)
                times_exp = time_vals.unsqueeze(0).unsqueeze(2).expand(
                    self.batch_size, -1, 1)
                h_hat = self._run_generator(generator, z, times_exp, device)
                x_real = recovery(h, times_exp)
                x_fake = recovery(h_hat, times_exp)
                y_fake = discriminator(x_fake, times_exp)
                y_real = discriminator(x_real, times_exp)

                loss_d = F.binary_cross_entropy_with_logits(
                    y_real, torch.ones_like(y_real)
                ) + F.binary_cross_entropy_with_logits(
                    y_fake, torch.zeros_like(y_fake)
                )

                if loss_d.item() > 0.15:
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    optimizer_d.step()

            # --- Generator update ---
            x_batch = self._get_batch(train_data, self.batch_size, device)
            coeffs = self._make_coeffs(time_vals, x_batch)
            z = torch.randn(self.batch_size, self.seq_len, self.latent_dim,
                            device=device)
            times_exp = time_vals.unsqueeze(0).unsqueeze(2).expand(
                self.batch_size, -1, 1)

            with torch.no_grad():
                h = embedder(time_vals, coeffs, final_index)

            h_hat = self._run_generator(generator, z, times_exp, device)
            x_fake = recovery(h_hat, times_exp)
            y_fake = discriminator(x_fake, times_exp)

            loss_g_u = F.binary_cross_entropy_with_logits(
                y_fake, torch.ones_like(y_fake))

            # Moment matching loss
            loss_g_v1 = torch.mean(torch.abs(
                torch.sqrt(torch.var(x_fake, 0) + 1e-6)
                - torch.sqrt(torch.var(x_batch, 0) + 1e-6)))
            loss_g_v2 = torch.mean(torch.abs(
                torch.mean(x_fake, 0) - torch.mean(x_batch, 0)))
            loss_g_v = loss_g_v1 + loss_g_v2

            loss_g = loss_g_u + 100 * loss_g_v

            optimizer_gs.zero_grad()
            loss_g.backward()
            optimizer_gs.step()

            if step % 500 == 0:
                print(f"  [GT-GAN] Phase 2 step {step}/{n_joint}, "
                      f"loss_d: {loss_d.item():.4f}, "
                      f"loss_g_u: {loss_g_u.item():.4f}, "
                      f"loss_g_v: {loss_g_v.item():.4f}")

        print("  [GT-GAN] Phase 2 done.")

        # Store trained models
        self._models = {
            "generator": generator,
            "recovery": recovery,
            "embedder": embedder,
        }

        t_end = time.time()
        self._train_time = t_end - t_start

        # Peak GPU memory
        try:
            self._peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        except Exception:
            self._peak_mem_mb = 0.0

    def sample(self, n: int) -> np.ndarray:
        """Generate *n* synthetic samples (n, seq_len, n_features)."""
        if not self._models:
            raise RuntimeError("Model not trained. Call fit() first.")

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        generator = self._models["generator"]
        recovery = self._models["recovery"]
        embedder = self._models["embedder"]

        generator.eval()
        recovery.eval()
        embedder.eval()

        time_vals = torch.arange(self.seq_len, dtype=torch.float32, device=device)
        times_exp = time_vals.unsqueeze(0).unsqueeze(2).expand(n, -1, 1)

        with torch.no_grad():
            z = torch.randn(n, self.seq_len, self.latent_dim, device=device)
            h_hat = self._run_generator(generator, z, times_exp, device)
            x_hat = recovery(h_hat, times_exp)

        return x_hat.cpu().numpy()

    def num_parameters(self) -> int:
        """Return total trainable parameters across all sub-networks."""
        total = 0
        for name, model in self._models.items():
            total += sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def train_time_sec(self) -> float:
        return self._train_time

    @property
    def peak_gpu_mem_mb(self) -> float:
        return self._peak_mem_mb

    # ── Internal helpers ────────────────────────────────────────────────────

    def _build_models(self, device):
        """Build embedder, recovery, generator, supervisor, discriminator."""
        batch_size = self.batch_size
        nf = self.n_features
        hd = self.hidden_dim
        ld = self.latent_dim

        # Embedder: NeuralCDE
        ode_func = FinalTanh(nf, hd, hd, 3).to(device)
        embedder = NeuralCDE(
            func=ode_func,
            input_channels=nf,
            hidden_channels=hd,
            output_channels=hd,
        ).to(device)

        # Recovery: Multi_Layer_ODENetwork
        recovery = _MultiLayerODENet(
            input_size=hd,
            hidden_size=hd,
            output_size=nf,
            num_layer=self.num_rec_layers,
            delta_t=0.5,
            last_activation='tanh',
        ).to(device)

        # Supervisor: GRU net
        supervisor = _Net(hd, hd, hd, 2).to(device)

        # Discriminator: Multi_Layer_ODENetwork
        discriminator = _MultiLayerODENet(
            input_size=nf,
            hidden_size=hd,
            output_size=1,
            num_layer=self.num_disc_layers,
            delta_t=0.5,
            last_activation='identity',
        ).to(device)

        # Generator: CNF (use the original library's builder via ctfp_tools)
        (
            _controldiffeq,
            run_model_fn,
            parse_arguments,
            build_model_tabular_nonlinear,
            set_cnf_options,
            create_regularization_fns,
            _count_cnf_params,
        ) = _import_gtgan_modules()

        import argparse
        parser = parse_arguments()
        known_args = [
            "--input_size", str(nf),
            "--latent_size", str(ld),
            "--dims", "32-64-64-32",
            "--num_blocks", "1",
            "--divergence_fn", "approximate",
            "--nonlinearity", "softplus",
            "--solver", "dopri5",
            "--atol", "1e-3",
            "--rtol", "1e-2",
            "--step_size", "0.1",
            "--time_length", "1.0",
            "--train_T", "True",
            "--kinetic-energy", "0.05",
            "--jacobian-norm2", "0.01",
            "--directional-penalty", "0.01",
        ]
        args = parser.parse_args(known_args)
        args.effective_shape = self.latent_dim

        reg_fns, reg_coeffs = create_regularization_fns(args)
        generator = build_model_tabular_nonlinear(
            args, args.effective_shape, regularization_fns=reg_fns).to(device)
        set_cnf_options(args, generator)

        # Store the run_model reference and args for later use
        self._run_model_fn = run_model_fn
        self._gen_args = args

        return {
            "embedder": embedder,
            "recovery": recovery,
            "generator": generator,
            "supervisor": supervisor,
            "discriminator": discriminator,
        }

    def _get_batch(self, data, batch_size, device):
        """Sample a mini-batch of (batch, seq_len, n_features)."""
        idx = torch.randperm(len(data))[:batch_size]
        batch = torch.from_numpy(data[idx.numpy()]).float().to(device)
        return batch

    def _make_coeffs(self, times, x_batch):
        """Build natural cubic spline coefficients for the batch."""
        import controldiffeq
        return controldiffeq.natural_cubic_spline_coeffs(times, x_batch)

    def _run_generator(self, generator, z, times, device):
        """Run the CNF generator: sample z -> generate h_hat.

        NOTE: The vendored run_latent_ctfp_model5_3 uses values.shape[1] as
        the latent dimension (a bug: it should be values.shape[2]). To work
        around this, we transpose z from (B, S, L) to (B, L, S) so that
        axis 1 (latent dim) = self.latent_dim matches what the CNF expects.
        """
        # Swap seq_len <-> latent_dim axes so values.shape[1] = latent_dim
        z = z.transpose(1, 2)  # (B, seq_len, latent_dim) -> (B, latent_dim, seq_len)
        return self._run_model_fn(
            self._gen_args, generator, z, times, device, z=True)
