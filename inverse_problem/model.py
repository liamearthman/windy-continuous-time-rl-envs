"""
Learned port-Hamiltonian model of the windy cart-pole.

Replaces the closed forms in dynamics.py with networks for the three physical
objects, keeping everything else fixed:

    H(q, p) = 1/2 p^T M_phi^-1(theta) p + V_phi(theta)
    f(z, u) = (J - R_phi(theta)) grad H(z) + G u
    sigma(z) = learned scalar diffusion on the p_theta channel

WHAT IS LEARNED AND WHAT IS NOT.  M^-1, V, D and the diffusion are learned.
J and G are NOT, and that is deliberate:

  * J is the symplectic form of the coordinates we chose, not a property of the
    plant.  Learning it would be learning our own bookkeeping.
  * G is known exactly -- u enters pdot_x with coefficient 1 -- and freezing it
    kills a gauge.  Under positions-only sensing the rescaling
    (M, p, V, D, G) -> (cM, cp, cV, cD, cG) leaves every observable unchanged,
    so a learned G would leave the overall mass scale unidentifiable.  This is
    also where the LieSPHGP paper's neural baseline failed: its learned M^-1 G
    had a SIGN ERROR near upright (0.57/-1.29/0.63 against a true 0.5/0.7/0.8),
    which flipped the controller.  We do not give the model that opportunity.

STRUCTURE BY CONSTRUCTION, NOT BY PENALTY.

  * M^-1 and D are symmetric positive definite via a Cholesky factor with
    exponentiated diagonal (LieSPHGP eq. 64).  det L > 0 for every input and
    every weight, so there is no regularizer to tune and no way to train into an
    indefinite mass matrix.
  * R touches the MOMENTUM BLOCK ONLY.  Rayleigh dissipation acts on velocities;
    a free 4x4 PSD matrix would put drag on the position rows, which is
    unphysical and gives the model somewhere to hide integrator error.
  * The nets see (cos theta, sin theta) and NOTHING ELSE.  Excluding x is the
    translation invariance that makes column 0 of the true Jacobian identically
    zero -- feed x in and the model can learn an x-dependence the data cannot
    refute.  The trig pair also makes every learned function automatically
    2*pi-periodic, which is correct for a pole and free.

GAUGE.  V is identifiable only up to an additive constant, so it is pinned at
V(theta_ref) = 0 by subtraction.  Compare learned and true potentials in the
same gauge or the comparison is meaningless.

INTEGRATION.  `heun_step` is the same arithmetic as dynamics.heun_step, which is
itself verified byte-equivalent to CartPoleEnv.integrator.  That is a
requirement, not a convenience: fitting a model through a different discrete map
than the one that generated the data pushes the discretization error into the
learned physics.  LieSPHGP uses a Lie-IMEX scheme instead, because a learned D
can be stiff enough to destabilize an explicit step -- worth revisiting if the
learned damping grows, but it would mean regenerating the data to match.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

N_STATE = 4
N_Q = 2


def _mlp(in_dim: int, out_dim: int, hidden: Sequence[int] = (64, 64)) -> nn.Sequential:
    """Tanh MLP.  Smooth activations matter here: grad H is taken by autograd and
    differentiated again for the parameter gradient, so a ReLU kink would put a
    discontinuity straight into the vector field."""
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def _zero_last_layer(net: nn.Sequential) -> None:
    """Start the network at exactly its level, with no initial ripple."""
    last = net[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


class SPDHead(nn.Module):
    """A 2x2 symmetric positive definite matrix as a function of features.

    Follows LieSPHGP B.1: a trainable level plus a network residual produce the
    three Cholesky entries, the two diagonal ones through exp.

        L = [[exp(a0),   0    ],
             [   a2,  exp(a1) ]],        out = L L^T

    Strictly positive definite for every input and every weight.  `init_diag`
    sets where the matrix starts: the network residual is zero-initialised, so
    training begins at exactly diag(init_diag^2).
    """

    def __init__(self, in_dim: int, init_diag: float = 1.0,
                 hidden: Sequence[int] = (64, 64)):
        super().__init__()
        self.net = _mlp(in_dim, 3, hidden)
        _zero_last_layer(self.net)
        self.level = nn.Parameter(
            torch.tensor([math.log(init_diag), math.log(init_diag), 0.0]))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        a = self.level + self.net(feat)                       # (B, 3)
        L = torch.zeros(*a.shape[:-1], 2, 2, dtype=a.dtype, device=a.device)
        L[..., 0, 0] = torch.exp(a[..., 0])
        L[..., 1, 1] = torch.exp(a[..., 1])
        L[..., 1, 0] = a[..., 2]
        return L @ L.transpose(-1, -2)


class DiffusionHead(nn.Module):
    """sigma(z) = softplus(g) + softplus(t) * sqrt(thetadot^2 + eps^2).

    Deliberately the same functional form as the plant rather than a free net:
    the wind has two scalars and the point of the first experiment is whether
    M, V and D are recoverable, not whether a network can fit a square root.
    Swap in an MLP on the smoothed |thetadot| when that becomes the question.
    Both scalars are non-negative by construction.
    """

    def __init__(self, init_gust: float = 1e-2, init_turb: float = 1e-2,
                 smooth_eps: float = 1e-3):
        super().__init__()
        self.raw_gust = nn.Parameter(torch.tensor(_inv_softplus(init_gust)))
        self.raw_turb = nn.Parameter(torch.tensor(_inv_softplus(init_turb)))
        self.smooth_eps = smooth_eps

    def forward(self, thetadot: torch.Tensor) -> torch.Tensor:
        v = torch.sqrt(thetadot**2 + self.smooth_eps**2)
        return nn.functional.softplus(self.raw_gust) + nn.functional.softplus(self.raw_turb) * v


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class PortHamiltonianModel(nn.Module):
    """The learned plant.  State order (x, theta, p_x, p_theta), as everywhere else."""

    def __init__(self, hidden: Sequence[int] = (64, 64), m_inv_init: float = 1.0,
                 d_init: float = 1e-2, learn_diffusion: bool = True,
                 theta_ref: float = 0.0, smooth_eps: float = 1e-3):
        super().__init__()
        self.m_inv_head = SPDHead(2, init_diag=m_inv_init, hidden=hidden)
        self.d_head = SPDHead(2, init_diag=d_init, hidden=hidden)
        self.v_net = _mlp(2, 1, hidden)
        self.diffusion_head = DiffusionHead(smooth_eps=smooth_eps) if learn_diffusion else None
        self.theta_ref = theta_ref

        # Fixed structure.  Buffers, not parameters -- these are never trained.
        J = torch.zeros(N_STATE, N_STATE, dtype=torch.float64)
        J[:N_Q, N_Q:] = torch.eye(N_Q, dtype=torch.float64)
        J[N_Q:, :N_Q] = -torch.eye(N_Q, dtype=torch.float64)
        self.register_buffer("J", J)
        self.register_buffer("G", torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64))
        self.register_buffer("G_noise", torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64))
        self.double()

    # -- the three learned objects -----------------------------------------

    @staticmethod
    def features(theta: torch.Tensor) -> torch.Tensor:
        """(cos, sin).  Periodic by construction, and x is deliberately absent."""
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    def M_inv(self, theta: torch.Tensor) -> torch.Tensor:
        return self.m_inv_head(self.features(theta))

    def D(self, theta: torch.Tensor) -> torch.Tensor:
        return self.d_head(self.features(theta))

    def potential(self, theta: torch.Tensor) -> torch.Tensor:
        """V, gauge-fixed to V(theta_ref) = 0."""
        v = self.v_net(self.features(theta)).squeeze(-1)
        ref = torch.as_tensor(self.theta_ref, dtype=theta.dtype, device=theta.device)
        v_ref = self.v_net(self.features(ref.reshape(1))).squeeze(-1)
        return v - v_ref

    # -- assembled dynamics -------------------------------------------------

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        theta, p = z[..., 1], z[..., N_Q:]
        Minv = self.M_inv(theta)
        kinetic = 0.5 * torch.einsum("...i,...ij,...j->...", p, Minv, p)
        return kinetic + self.potential(theta)

    def grad_hamiltonian(self, z: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        """grad_z H by autograd, keeping the graph.

        create_graph=True is load-bearing: the loss backpropagates THROUGH this
        gradient to reach the network weights.  Detaching here trains V and M to
        nothing, which the LieSPHGP paper flags explicitly for the same reason.
        """
        z = z if z.requires_grad else z.detach().requires_grad_(True)
        H = self.hamiltonian(z).sum()
        (g,) = torch.autograd.grad(H, z, create_graph=create_graph)
        return g

    def dissipation(self, theta: torch.Tensor) -> torch.Tensor:
        """R, (B, 4, 4), zero outside the momentum block."""
        D = self.D(theta)
        R = torch.zeros(*D.shape[:-2], N_STATE, N_STATE, dtype=D.dtype, device=D.device)
        R[..., N_Q:, N_Q:] = D
        return R

    def drift(self, z: torch.Tensor, u: torch.Tensor,
              create_graph: bool = True) -> torch.Tensor:
        """f = (J - R) grad H + G u."""
        gH = self.grad_hamiltonian(z, create_graph=create_graph)
        R = self.dissipation(z[..., 1])
        return torch.einsum("ij,...j->...i", self.J, gH) \
            - torch.einsum("...ij,...j->...i", R, gH) \
            + self.G * u.unsqueeze(-1)

    def diffusion(self, z: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        """Scalar sigma(z).  Depends on thetadot = (M^-1 p)_1, so it is coupled
        to the learned mass matrix -- wind and mechanics are not independent."""
        if self.diffusion_head is None:
            return torch.zeros(z.shape[:-1], dtype=z.dtype, device=z.device)
        thetadot = torch.einsum("...ij,...j->...i", self.M_inv(z[..., 1]), z[..., N_Q:])[..., 1]
        return self.diffusion_head(thetadot)

    # -- the prediction operator -------------------------------------------

    def heun_step(self, z: torch.Tensor, u: torch.Tensor, dt: float,
                  dW: torch.Tensor | None = None,
                  create_graph: bool = True) -> torch.Tensor:
        """One stochastic-Heun step.  Same arithmetic as dynamics.heun_step.

        dW=None means zero, which is the TRAINING call: the model does not know
        the realized wind, so it predicts the conditional mean.  (Strictly, the
        mean of the Heun map is not the map at dW=0 -- the corrector is nonlinear
        in dW, leaving an O(dt) bias -- which is the Ito/Stratonovich gap showing
        up again.  Small at these sigma; the EKF objective accounts for it.)
        """
        f0 = self.drift(z, u, create_graph)
        if dW is None:
            z_t = z + f0 * dt
            f1 = self.drift(z_t, u, create_graph)
            return z + 0.5 * (f0 + f1) * dt
        s0 = self.diffusion(z, create_graph)
        z_t = z + f0 * dt + (s0 * dW).unsqueeze(-1) * self.G_noise
        s1 = self.diffusion(z_t, create_graph)
        f1 = self.drift(z_t, u, create_graph)
        return z + 0.5 * (f0 + f1) * dt + (0.5 * (s0 + s1) * dW).unsqueeze(-1) * self.G_noise


# ---------------------------------------------------------------------------
# Ground truth in the model's own interface, for evaluation.
# ---------------------------------------------------------------------------

class TrueModel:
    """The analytic plant exposed through PortHamiltonianModel's methods.

    Lets the evaluation compare like with like -- M^-1 against M^-1, V against V
    in the same gauge -- rather than comparing a network to a Params object.
    """

    def __init__(self, p, theta_ref: float = 0.0):
        self.p, self.theta_ref = p, theta_ref

    def M_inv(self, theta: torch.Tensor) -> torch.Tensor:
        p = self.p
        c = torch.cos(theta)
        det = (4.0 / 3.0) * p.beta * p.m_tot - (p.alpha * c) ** 2
        out = torch.zeros(*theta.shape, 2, 2, dtype=theta.dtype)
        out[..., 0, 0] = (4.0 / 3.0) * p.beta / det
        out[..., 0, 1] = -p.alpha * c / det
        out[..., 1, 0] = -p.alpha * c / det
        out[..., 1, 1] = p.m_tot / det
        return out

    def potential(self, theta: torch.Tensor) -> torch.Tensor:
        ref = torch.as_tensor(self.theta_ref, dtype=theta.dtype)
        return self.p.gamma * (torch.cos(theta) - torch.cos(ref))

    def D(self, theta: torch.Tensor) -> torch.Tensor:
        p = self.p
        b_pole = p.b_base + p.b_amp * torch.sqrt(torch.sin(theta) ** 2 + p.smooth_eps ** 2)
        out = torch.zeros(*theta.shape, 2, 2, dtype=theta.dtype)
        out[..., 0, 0] = p.b_cart
        out[..., 1, 1] = b_pole
        return out
