"""PSIN model implementation.

This implementation follows the experiments reported in the manuscript. The
operator stored in ``self.W`` is used in the implementation orientation. In the
paper notation, this operation corresponds to the wavelet-induced projection
term in the unfolded update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR


@dataclass
class TrainingHistory:
    """Training statistics returned by PSIN."""

    epoch_losses: List[float]
    trace_losses: List[float]


class PSIN(nn.Module):
    """Physics-informed self-supervised ISTA-unrolled network.

    Parameters
    ----------
    seismic_data:
        Normalized seismic section with shape ``(n_samples, n_traces)``.
    init_operator:
        Toeplitz-Ricker initialization matrix with shape ``(n_samples, n_samples)``.
    initial_sparse_state:
        Initial hidden sparse representation with shape ``(n_samples, n_traces)``.
    n_traces_train:
        Number of traces used during training. This can be smaller than the
        number of traces in ``seismic_data`` for quick tests.
    n_stages:
        Number of weakly unrolled sparse-refinement stages.
    lr:
        Initial learning rate.
    tau:
        Fixed soft-thresholding parameter.
    """

    def __init__(
        self,
        seismic_data: np.ndarray,
        init_operator: np.ndarray,
        initial_sparse_state: np.ndarray,
        n_traces_train: Optional[int] = None,
        n_stages: int = 3,
        lr: float = 1e-4,
        tau: float = 0.001,
        scheduler_tmax: int = 10,
        scheduler_eta_min: float = 1e-6,
    ) -> None:
        super().__init__()

        if seismic_data.ndim != 2:
            raise ValueError("seismic_data must be a 2D array")
        if init_operator.shape[0] != init_operator.shape[1]:
            raise ValueError("init_operator must be square")
        if init_operator.shape[0] != seismic_data.shape[0]:
            raise ValueError("init_operator must match the sample dimension of seismic_data")
        if initial_sparse_state.shape != seismic_data.shape:
            raise ValueError("initial_sparse_state must have the same shape as seismic_data")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seismic_data = np.asarray(seismic_data, dtype=np.float32)
        self.middle_state = np.asarray(initial_sparse_state, dtype=np.float32).copy()
        self.n_samples, self.n_traces = self.seismic_data.shape
        self.n_traces_train = int(n_traces_train or self.n_traces)
        self.n_stages = int(n_stages)
        self.tau = float(tau)

        self.W = nn.Parameter(torch.tensor(init_operator, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.beta_raw = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam([self.W, self.alpha, self.beta_raw], lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=scheduler_tmax, eta_min=scheduler_eta_min)
        self.to(self.device)

    @staticmethod
    def soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Apply elementwise soft-thresholding."""
        return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)

    @staticmethod
    def stable_operator_correlation(operator: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Build the spectrally normalized operator-correlation feedback matrix.

        Gradients are stopped through the quadratic branch to prevent unstable
        higher-order coupling with the direct projection pathway.
        """
        operator_detached = operator.detach()
        correlation = operator_detached @ operator_detached.T
        spectral_norm = torch.linalg.norm(correlation, ord=2)
        return correlation / (spectral_norm + eps)

    def forward(self, trace: torch.Tensor, initial_state: torch.Tensor) -> torch.Tensor:
        """Run the weakly unrolled sparse-inference process for one trace."""
        feedback_operator = self.stable_operator_correlation(self.W)
        beta = -F.softplus(self.beta_raw)
        state = initial_state

        for _ in range(self.n_stages):
            hidden = self.alpha * (self.W @ trace) + state + beta * (feedback_operator @ state)
            state = self.soft_threshold(hidden, threshold=self.tau)

        return state

    def train_one_trace(self, trace_id: int) -> float:
        """Train PSIN on one trace and update the external memory state."""
        trace = torch.tensor(self.seismic_data[:, trace_id], dtype=torch.float32, device=self.device).view(-1, 1)
        initial_state = torch.tensor(self.middle_state[:, trace_id], dtype=torch.float32, device=self.device).view(-1, 1)

        estimated_reflectivity = self.forward(trace, initial_state)
        reconstructed_trace = self.W @ estimated_reflectivity
        loss = self.loss_function(reconstructed_trace, trace)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([self.W], max_norm=1.0)
        self.optimizer.step()

        self.middle_state[:, trace_id] = estimated_reflectivity.detach().cpu().numpy().ravel()
        return float(loss.item())

    def fit(self, epochs: int = 20, verbose: bool = True) -> TrainingHistory:
        """Train the model for a fixed number of epochs."""
        epoch_losses: List[float] = []
        trace_losses: List[float] = []

        for epoch in range(int(epochs)):
            total_loss = 0.0
            for trace_id in range(self.n_traces_train):
                loss = self.train_one_trace(trace_id)
                total_loss += loss
                trace_losses.append(loss)
                if verbose:
                    print(f"[epoch {epoch + 1:03d}] trace {trace_id:04d}, loss={loss:.6f}")

            self.scheduler.step()
            mean_loss = total_loss / self.n_traces_train
            epoch_losses.append(mean_loss)
            if verbose:
                print(f"Epoch {epoch + 1:03d} mean loss: {mean_loss:.6f}")

        return TrainingHistory(epoch_losses=epoch_losses, trace_losses=trace_losses)

    def save_reflectivity_mat(self, path: str = "output_reflectivity.mat") -> None:
        """Save the estimated sparse reflectivity-like representation."""
        sio.savemat(path, {"output_ref_matrix": self.middle_state})

    def get_operator(self) -> np.ndarray:
        """Return the current trainable operator as a NumPy array."""
        return self.W.detach().cpu().numpy()
