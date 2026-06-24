import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..registry import register_env, EnvSpec, PhysicsPriors
from ..losses import compute_sequence_loss
from ..metrics import (
    angle_mse,
    autoregressive_rollout,
    extract_damping_diag,
    extract_phast_stiffness_metrics,
    r2_score,
    wrap_to_pi,
)

# need to implement control



class PendulumParams:
    """Physical parameters for pendulum."""
    m: float = 1.0       # mass
    L: float = 1.0       # length
    g: float = 9.81      # gravity
    dt: float = 0.05     # time step


class PendulumSimulator:
    """
    Pendulum simulator with configurable damping.

    State: x = [theta, p] where p = m*L^2 * omega (angular momentum)

    Dynamics (port-Hamiltonian form):
        dtheta/dt = p / I
        dp/dt = -m*g*L*sin(theta) - D(theta) * (p / I)

    where I = m*L^2 is the moment of inertia.
    """

    def __init__(
        self,
        params: PendulumParams = None,
        damping_type: str = 'none',  # 'none', 'constant', 'windy'
        damping_coeff: float = 0.0,  # For constant damping
        b_base: float = 0.3,         # For windy: base damping
        b_amplitude: float = 0.5,    # For windy: amplitude of angle-dependent part
    ):
        self.params = params or PendulumParams()
        self.damping_type = damping_type
        self.damping_coeff = damping_coeff
        self.b_base = b_base
        self.b_amplitude = b_amplitude

        # Derived quantities
        self.I = self.params.m * self.params.L**2  # moment of inertia

    def compute_damping(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute damping coefficient D(theta).

        Returns:
            D: Damping coefficient (same shape as theta)
        """
        if self.damping_type == 'none':
            return torch.zeros_like(theta)
        elif self.damping_type == 'constant':
            return torch.full_like(theta, self.damping_coeff)
        elif self.damping_type == 'windy':
            # D(theta) = b_base + b_amplitude * |sin(theta)|
            # Physical: air resistance is higher when pendulum is horizontal
            return self.b_base + self.b_amplitude * torch.abs(torch.sin(theta))
        else:
            raise ValueError(f"Unknown damping type: {self.damping_type}")

    def compute_energy(self, theta: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute Hamiltonian H(theta, p) = T + V.

        T = p^2 / (2*I)           (kinetic energy)
        V = m*g*L*(1 - cos(theta)) (potential energy, zero at bottom)
        """
        T = p**2 / (2 * self.I)
        V = self.params.m * self.params.g * self.params.L * (1 - torch.cos(theta))
        return T + V

    def compute_dissipation_rate(self, theta: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute energy dissipation rate: dH/dt = -v^T D(theta) v

        where v = p / I is the angular velocity.

        Returns:
            dissipation: dH/dt (should be <= 0 for passive systems)
        """
        v = p / self.I
        D = self.compute_damping(theta)
        return -D * v**2
    
    # u is the torque input
    # need to get torque input u from the action the RL agent chooses. 
    
    def step(
        self,
        theta: torch.Tensor,
        p: torch.Tensor,
        u: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate one time step using Strang splitting:
            half-D -> Leapfrog -> half-D

        This preserves the port-Hamiltonian structure for the unforced dynamics.
        When a control input u is provided (zero-order hold over the step), it is
        injected additively into the momentum equation (torque/force input).
        """
        dt = self.params.dt
        m, L, g = self.params.m, self.params.L, self.params.g
        if u is None:
            u = torch.zeros_like(theta)

        # Step 1: Half damping at current theta
        D = self.compute_damping(theta)
        v = p / self.I
        p = p - 0.5 * dt * D * v

        # Step 2: Leapfrog (symplectic for conservative part)
        # Half momentum kick
        p_half = p - 0.5 * dt * m * g * L * torch.sin(theta) + 0.5 * dt * u
        # Full position update
        theta_new = theta + dt * p_half / self.I
        # Half momentum kick at new position
        p = p_half - 0.5 * dt * m * g * L * torch.sin(theta_new) + 0.5 * dt * u

        # Step 3: Half damping at new theta
        D_new = self.compute_damping(theta_new)
        v = p / self.I
        p = p - 0.5 * dt * D_new * v

        return theta_new, p
