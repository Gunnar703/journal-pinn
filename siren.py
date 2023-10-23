import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from typing import Callable
from random import random


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.last_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def system_ode_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    m: torch.Tensor,
    c: torch.Tensor,
    k: torch.Tensor,
    u_max: float,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
):
    """Computes loss due to the residual

    M y_xx + C y_x + K y - f / u_max == 0
    """
    y_x, y_xx = y * 0, y * 0
    for dim in range(y_x.shape[1]):
        y_x[:, dim] = gradient(y[:, dim], x).squeeze()
        y_xx[:, dim] = gradient(y_x[:, dim], x).squeeze()

    residual = torch.mm(m, y_xx.T) + torch.mm(c, y_x.T) + torch.mm(k, y.T) - (f / u_max)
    return loss_fn(residual, residual * 0)


def latin_hypercube_1D(t0, t1, n, device="cuda"):
    endpts = torch.linspace(t0, t1, n + 1, device=device)
    sample_pts = torch.as_tensor(
        [random() * (endpts[i + 1] - endpts[i]) + endpts[i] for i in range(n)],
        device=device,
    )
    return sample_pts


def get_4_dof_model(
    hidden_layers: int = 3,
    hidden_features: int = 32,
    first_omega_0: float = 30.0,
    hidden_omega_0: float = 30.0,
    device: str = "cuda",
):
    out_features = 4
    model = Siren(
        in_features=1,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        outermost_linear=True,
        first_omega_0=first_omega_0,
        hidden_omega_0=hidden_omega_0,
    ).to(device)
    return model


class SineLayer(nn.Module):
    """
    Adapted from
    https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb
    """

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


class Siren(nn.Module):
    """
    Adapted from
    https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb
    """

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        output = self.net(coords)

        return output, coords
