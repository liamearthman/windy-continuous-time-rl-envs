"""
Forward model for windy-cart-pole parameter recovery.

This is the same map as `environment.cartpole.CartPoleEnv.integrator`, re-expressed
as pure functions of (state, input, parameters, noise) so that it can be
differentiated.  Two things about the env version make it unusable for an inverse
problem directly:

  * it reads the physical parameters off `self`, so they are not differentiable
    inputs, and
  * `integrator()` draws dW from `self._np_random` internally, so there is no way
    to hold a noise realization fixed.

Pathwise gradients need both fixed, hence this module.  `test_dynamics.py` asserts
agreement with the env to machine precision, so the two cannot drift apart silently.

COORDINATES.  The dynamics depend on the four physical parameters (m_c, m_p, l, g)
only through four combinations,

    m_tot = m_c + m_p,   alpha = m_p*l,   beta = m_p*l^2,   gamma = m_p*g*l,

because the mass matrix is [[m_tot, alpha*cos(th)], [alpha*cos(th), (4/3)*beta]]
and the pole-momentum drift is gamma*sin(th) - alpha*sin(th)*xdot*thdot.  Nothing
else appears.  The map back is a bijection,

    m_p = alpha^2/beta,   l = beta/alpha,   g = gamma/alpha,   m_c = m_tot - m_p,

so all four physical parameters are structurally identifiable -- but note that g is
recovered as a RATIO of two estimated quantities and m_c only as a DIFFERENCE, so
they inherit the errors of alpha and m_p respectively.  Fit in combo coordinates,
convert at the end, and report conditioning in both.  Optimizing the physical
parameters directly is the same problem through a worse-conditioned change of basis.

WIND.  The diffusion is sigma(z) = sigma_gust + sigma_turb*sqrt(thdot^2 + eps^2),
delivered through G = (0,0,0,1) -- the pole's momentum only.  This makes the noise
RANK ONE in a 4-dimensional state: three of the four increments are deterministic
given (z, u).  There is no non-degenerate transition density to write down, and a
likelihood has to be built on the p_theta increment conditional on the other three.
See the `b`/`sigma` Jacobians below: the mechanical parameters enter sigma only
through thdot, which is why mechanical and wind parameters are not separable by
inspection even though the drift does not contain sigma at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np

# Order is load-bearing: it fixes the column order of `drift_jac_params`,
# `diffusion_jac_params`, and the optimizer's parameter vector.
PARAM_NAMES: Tuple[str, ...] = (
    "m_tot", "alpha", "beta", "gamma",          # H: mass matrix and potential
    "b_cart", "b_base", "b_amp",                # R: damping
    "sigma_gust", "sigma_turb",                 # wind
)

# Index groups.  Each names a block that can be fitted or frozen on its own --
# the damping block is the one a D-network replaces, and the wind block is the
# only one the drift does NOT depend on (those columns of `drift_jac_params`
# are structurally zero).
MECHANICAL = (0, 1, 2, 3)
DAMPING = (4, 5, 6)
WIND = (7, 8)

N_STATE = 4
N_PARAM = 9


@dataclass(frozen=True)
class Params:
    """Parameters in combo coordinates.  `smooth_eps` is fixed, not estimated."""

    m_tot: float
    alpha: float
    beta: float
    gamma: float
    b_cart: float = 0.0
    b_base: float = 0.0
    b_amp: float = 0.0
    sigma_gust: float = 0.0
    sigma_turb: float = 0.0
    smooth_eps: float = 1e-3

    # -- conversions ----------------------------------------------------------
    @classmethod
    def from_physical(cls, masscart, masspole, length, gravity,
                      sigma_gust=0.0, sigma_turb=0.0, smooth_eps=1e-3,
                      b_cart=0.0, b_base=0.0, b_amp=0.0) -> "Params":
        return cls(
            m_tot=masscart + masspole,
            alpha=masspole * length,
            beta=masspole * length**2,
            gamma=masspole * gravity * length,
            b_cart=b_cart,
            b_base=b_base,
            b_amp=b_amp,
            sigma_gust=sigma_gust,
            sigma_turb=sigma_turb,
            smooth_eps=smooth_eps,
        )

    @classmethod
    def from_env(cls, env) -> "Params":
        """Read the combo parameters off a CartPoleEnv instance."""
        return cls.from_physical(
            env.masscart, env.masspole, env.length, env.gravity,
            env.sigma_gust, env.sigma_turb, env.smooth_eps,
            env.b_cart, env.b_base, env.b_amplitude,
        )

    def to_physical(self) -> dict:
        """Back to (m_c, m_p, l, g).  The damping parameters are already
        physical -- they are not lumped, so they pass through unchanged."""
        masspole = self.alpha**2 / self.beta
        return {
            "masspole": masspole,
            "length": self.beta / self.alpha,
            "gravity": self.gamma / self.alpha,
            "masscart": self.m_tot - masspole,
            "b_cart": self.b_cart,
            "b_base": self.b_base,
            "b_amp": self.b_amp,
        }

    # -- flat-vector interface for the optimizer ------------------------------
    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in PARAM_NAMES], dtype=float)

    @classmethod
    def from_vector(cls, v, smooth_eps: float = 1e-3) -> "Params":
        return cls(*(float(x) for x in np.asarray(v, dtype=float)), smooth_eps=smooth_eps)

    def perturbed(self, index: int, delta: float) -> "Params":
        """Copy with one parameter nudged -- for finite-difference checks."""
        name = PARAM_NAMES[index]
        return replace(self, **{name: getattr(self, name) + delta})


# ---------------------------------------------------------------------------
# Core physics.  Written in terms of an array module `xp` and returning plain
# tuples of scalars so that the SAME source works for numpy and for torch
# tensors (the autodiff cross-check).  Nothing here allocates an array, so
# there is no np.array/torch.stack fork in the physics itself.
# ---------------------------------------------------------------------------

def _det(cos_th, p: Params):
    return (4.0 / 3.0) * p.beta * p.m_tot - (p.alpha * cos_th) ** 2


def velocities_raw(theta, px, pth, p: Params, xp=np):
    """(xdot, thdot) = M(theta)^-1 @ (px, pth)."""
    c = xp.cos(theta)
    det = _det(c, p)
    xdot = ((4.0 / 3.0) * p.beta * px - p.alpha * c * pth) / det
    thdot = (-p.alpha * c * px + p.m_tot * pth) / det
    return xdot, thdot


def momenta_raw(theta, xdot, thdot, p: Params, xp=np):
    """Inverse of `velocities_raw`: (px, pth) = M(theta) @ (xdot, thdot)."""
    c = xp.cos(theta)
    return (p.m_tot * xdot + p.alpha * c * thdot,
            p.alpha * c * xdot + (4.0 / 3.0) * p.beta * thdot)


def damping_raw(theta, p: Params, xp=np):
    """(b_cart, b_pole(theta)) -- the diagonal of R's momentum block.

    |sin(theta)| is smoothed to sqrt(sin^2 + eps^2) so that dR/dtheta exists at
    upright and hanging.  Must match CartPoleEnv.compute_damping exactly.
    """
    b_pole = p.b_base + p.b_amp * xp.sqrt(xp.sin(theta) ** 2 + p.smooth_eps ** 2)
    return p.b_cart, b_pole


def drift_raw(z, u, p: Params, xp=np):
    """The drift f(z, u; theta) as a 4-tuple, state order (x, theta, px, pth)."""
    _, theta, px, pth = z[0], z[1], z[2], z[3]
    s = xp.sin(theta)
    xdot, thdot = velocities_raw(theta, px, pth, p, xp)
    b_c, b_p = damping_raw(theta, p, xp)
    return (xdot,
            thdot,
            u - b_c * xdot,
            p.gamma * s - p.alpha * s * xdot * thdot - b_p * thdot)


def diffusion_raw(z, p: Params, xp=np):
    """sigma(z) = sigma_gust + sigma_turb*sqrt(thdot^2 + eps^2), a scalar."""
    _, thdot = velocities_raw(z[1], z[2], z[3], p, xp)
    return p.sigma_gust + p.sigma_turb * xp.sqrt(thdot**2 + p.smooth_eps**2)


# -- numpy-packing wrappers --------------------------------------------------

def velocities(z, p: Params) -> np.ndarray:
    return np.array(velocities_raw(z[1], z[2], z[3], p), dtype=float)


def drift(z, u, p: Params) -> np.ndarray:
    """u is taken as already clipped; the env clips to +/-max_force before use."""
    return np.array(drift_raw(z, u, p), dtype=float)


def diffusion(z, p: Params) -> float:
    return float(diffusion_raw(z, p))


# The noise direction: wind hits the pole's momentum only.
G_NOISE = np.array([0.0, 0.0, 0.0, 1.0])


def heun_step(z, u, dW, dt, p: Params) -> np.ndarray:
    """One stochastic-Heun step with the noise increment supplied, not drawn.

    Identical arithmetic to CartPoleEnv.integrator: one dW reused by predictor
    and corrector, trapezoidal in BOTH drift and diffusion.  That reuse is what
    makes the scheme converge to the Stratonovich solution rather than the Ito
    one -- which matters when writing a likelihood, not for pathwise gradients,
    where all that is required is that this map be the one the data came from.
    """
    z = np.asarray(z, dtype=float)
    s0 = diffusion(z, p)
    f0 = drift(z, u, p)
    z_t = z + f0 * dt + s0 * dW * G_NOISE
    s1 = diffusion(z_t, p)
    f1 = drift(z_t, u, p)
    return z + 0.5 * (f0 + f1) * dt + 0.5 * (s0 + s1) * dW * G_NOISE


# ---------------------------------------------------------------------------
# Port-Hamiltonian structure.
#
# The cart-pole in these coordinates is not merely pH-representable -- it is
# CANONICAL.  Verified symbolically (sympy, exact zero, not a tolerance):
#
#     J @ grad H(z) + G_INPUT * u  ==  drift(z, u)            identically
#
# with H = 1/2 p^T M(theta)^-1 p + gamma*cos(theta) and J the canonical
# symplectic form below.  Two consequences worth stating plainly:
#
# R = 0.  There is NO dissipation in this model -- the gym cart-pole rides a
# frictionless track and `_drift` contains no damping term.  So the professor's
# target form (J - R) grad H + G u degenerates here to J grad H + G u, and the
# deterministic system is LOSSLESS: grad H^T J grad H = 0 (J skew), hence
# Hdot = u * xdot, power in through the cart port alone.  Contrast the windy
# PENDULUM, which carries R = diag(0, D(theta)) with D(theta) > 0.  Any apparent
# damping recovered from this cart-pole data is an artifact -- of the
# integrator, of observation noise, or of a misspecified model.
#
# TWO PORTS, NOT ONE.  The wind enters through G_NOISE = e_4, which is the port
# collocated with theta: grad_{p_theta} H = thetadot.  So the full model is a
# stochastic pH system with an input port (force on the cart) and a noise port
# (torque on the pole):
#
#     dz = [J grad H + G_INPUT u] dt + G_NOISE sigma(z) dW
#
# In STRATONOVICH form -- which is the form stochastic Heun actually converges
# to -- the classical chain rule holds, so the energy balance is exactly the
# sum of the two ports with no correction term:
#
#     dH = u xdot dt + sigma(z) thetadot dW
#
# This is the strongest argument for keeping the Heun/Stratonovich convention
# rather than "fixing" it to Ito: in Ito form the same balance picks up a
# spurious-looking +1/2 sigma^2 (M^-1)_22 dt, the rate at which wind pumps
# energy into the pole.  That term is real physics, but it is a correction
# rather than a port, and it makes the passivity argument harder to state.
#
# CAVEAT for long horizons.  Stochastic Heun is NOT symplectic.  On a lossless
# system that means secular energy drift, O(dt^2) per step and accumulating
# over the horizon, which an estimator can absorb into the parameters as
# fictitious dissipation.  `data.energy_drift` measures it so the bias is
# visible rather than inferred.  (PendulumEnv's Strang splitting IS symplectic
# and would bound it -- that is the real trade against Heun, not accuracy.)
# ---------------------------------------------------------------------------

# Canonical symplectic form for state order (x, theta, p_x, p_theta).
J_PH = np.block([
    [np.zeros((2, 2)), np.eye(2)],
    [-np.eye(2), np.zeros((2, 2))],
])

# Input port: the force acts on the cart, i.e. on p_x.
G_INPUT = np.array([0.0, 0.0, 1.0, 0.0])


def hamiltonian_raw(z, p: Params, xp=np):
    """H = 1/2 p^T M(theta)^-1 p + gamma*cos(theta), the total energy.

    V = gamma*cos(theta) is maximal upright (theta = 0) and minimal hanging
    (theta = pi), which is the sign convention `_drift` already implies.
    """
    theta, px, pth = z[1], z[2], z[3]
    xdot, thdot = velocities_raw(theta, px, pth, p, xp)
    return 0.5 * (px * xdot + pth * thdot) + p.gamma * xp.cos(theta)


def grad_hamiltonian_raw(z, p: Params, xp=np):
    """grad H as a 4-tuple.  Closed form, verified symbolically against
    sympy's derivative of `hamiltonian_raw` and against torch autograd.

    dH/dx = 0 (translation invariance), and the momentum half is just the
    velocities -- which is why `drift`'s first two rows and `grad H`'s last two
    are the same numbers.
    """
    theta, px, pth = z[1], z[2], z[3]
    s = xp.sin(theta)
    xdot, thdot = velocities_raw(theta, px, pth, p, xp)
    return (0.0 * theta,                                  # keeps the dtype/shape
            p.alpha * s * xdot * thdot - p.gamma * s,
            xdot,
            thdot)


def hamiltonian(z, p: Params) -> float:
    return float(hamiltonian_raw(z, p))


def grad_hamiltonian(z, p: Params) -> np.ndarray:
    return np.array(grad_hamiltonian_raw(z, p), dtype=float)


def dissipation_matrix(z, p: Params) -> np.ndarray:
    """R, shape (4, 4).  Zero everywhere but the momentum diagonal.

    This block structure is a modelling commitment, not a convenience: Rayleigh
    dissipation acts on velocities, so R touches the momentum block only.  A
    D-network must be constrained the same way -- a free 4x4 PSD matrix would
    put drag on the position rows, which is unphysical and gives the net a place
    to hide integrator error.
    """
    b_c, b_p = damping_raw(z[1], p)
    R = np.zeros((N_STATE, N_STATE))
    R[2, 2] = b_c
    R[3, 3] = b_p
    return R


def drift_from_ph(z, u, p: Params) -> np.ndarray:
    """The drift assembled from pH structure instead of written out:

        f = (J - R(z)) grad H(z) + G u

    Equal to `drift` identically.  Kept because it is the form that generalizes:
    when M, V and D stop being closed forms and get learned, this is the line
    that stays and `drift` is the one that goes away.
    """
    return (J_PH - dissipation_matrix(z, p)) @ grad_hamiltonian(z, p) + G_INPUT * u
