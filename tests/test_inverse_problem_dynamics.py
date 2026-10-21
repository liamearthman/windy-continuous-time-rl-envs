# tests/test_inverse_problem_dynamics.py
"""
Agreement tests for the windy-cart-pole forward model in inverse_problem/.

Two kinds of claim are checked here:

  1. `inverse_problem.dynamics` is the SAME MAP as environment/cartpole.py --
     velocities, drift, diffusion and one Heun step agree to a few ulp.  Not bit
     exact, and deliberately not asserted to be: the combo coordinates form the
     same products in a different order ((4/3)*beta*m_tot against the env's
     m_tot*(4/3)*m_p*l**2), which moves the last bit.  The tolerances below are
     a few ulp of the quantities involved -- tight enough that any real
     disagreement in the physics fails them.
     If someone edits the env's physics these fail, which is the point: a
     recovered parameter is meaningless if the forward model used to recover it
     is not the model that generated the data.

  2. The torch backend agrees with the numpy one, and the drift is
     differentiable through it.  Autodiff is now the ONLY route to a gradient
     here -- the analytic Jacobians were deleted along with the adjoint plan --
     so the torch path is load-bearing and gets its own coverage.
"""

import unittest
from dataclasses import replace

import numpy as np
import torch

from environment.cartpole import CartPoleEnv
from inverse_problem import dynamics as dyn
from inverse_problem.dynamics import PARAM_NAMES, Params

# A state away from every special point: theta neither 0 nor pi/2, both momenta
# nonzero and opposite in sign, so the Coriolis term and both mass-matrix
# off-diagonals are active.  Degeneracies hide at the symmetric states, which is
# why those get their own test below rather than being used as the default.
Z0 = np.array([0.3, 0.4, -0.7, 0.25])
U0 = 2.5
DT = 0.02

# Agreement with the env: a few ulp, not bit-exact.  See the note above.
RTOL_ENV = 1e-13
ATOL_ENV = 1e-14



class InverseProblemDynamicsTest(unittest.TestCase):
    def setUp(self):
        self.env = CartPoleEnv()
        self.env.reset(seed=0)
        self.env_params = Params.from_env(self.env)
        # Wind turned up from the 0.002/0.001 defaults so that the
        # sigma-dependent terms are numerically visible to finite differences.
        self.params = replace(self.env_params, sigma_gust=0.05, sigma_turb=0.02)

    # -- coordinates --------------------------------------------------------

    def test_physical_roundtrip(self):
        phys = dict(masscart=1.0, masspole=0.1, length=0.5, gravity=9.8)
        back = Params.from_physical(**phys).to_physical()
        for k, v in phys.items():
            self.assertAlmostEqual(back[k], v, places=12, msg=k)

    def test_combo_values_match_env(self):
        p, e = self.env_params, self.env
        self.assertAlmostEqual(p.m_tot, e.masscart + e.masspole, places=15)
        self.assertAlmostEqual(p.alpha, e.polemass_length, places=15)  # m_p * l
        self.assertAlmostEqual(p.beta, e.masspole * e.length ** 2, places=15)
        self.assertAlmostEqual(p.gamma, e.masspole * e.gravity * e.length, places=15)

    def test_vector_roundtrip(self):
        v = self.params.to_vector()
        self.assertEqual(list(v), [getattr(self.params, n) for n in PARAM_NAMES])
        self.assertEqual(Params.from_vector(v, self.params.smooth_eps), self.params)

    def test_momenta_inverts_velocities(self):
        xdot, thdot = dyn.velocities_raw(Z0[1], Z0[2], Z0[3], self.params)
        px, pth = dyn.momenta_raw(Z0[1], xdot, thdot, self.params)
        self.assertAlmostEqual(px, Z0[2], places=14)
        self.assertAlmostEqual(pth, Z0[3], places=14)

    # -- same map as the environment ----------------------------------------

    def test_velocities_match_env(self):
        mine = dyn.velocities(Z0, self.env_params)
        theirs = np.asarray(self.env._velocities(Z0[1], Z0[2], Z0[3]))
        np.testing.assert_allclose(mine, theirs, rtol=RTOL_ENV, atol=ATOL_ENV)

    def test_drift_matches_env(self):
        np.testing.assert_allclose(
            dyn.drift(Z0, U0, self.env_params),
            self.env._drift(Z0, U0), rtol=RTOL_ENV, atol=ATOL_ENV)

    def test_diffusion_matches_env(self):
        self.assertAlmostEqual(
            dyn.diffusion(Z0, self.env_params), self.env.diffusion(Z0), places=13)

    def test_heun_step_matches_env(self):
        """Inject a known dW on both sides -- the env draws it internally."""
        xi = 0.8723

        class FixedRng:
            def standard_normal(self):
                return xi

        self.env.state = Z0.copy()
        self.env._np_random = FixedRng()
        theirs = np.asarray(self.env.integrator(np.array([U0]), DT))

        mine = dyn.heun_step(Z0, U0, np.sqrt(DT) * xi, DT, self.env_params)
        np.testing.assert_allclose(mine, theirs, rtol=RTOL_ENV, atol=ATOL_ENV)

    def test_drift_and_step_match_env_with_damping_on(self):
        """Damping defaults to zero, so every other test runs with R = 0.

        This one turns it on in the env and checks both sides still agree --
        otherwise the damping code is only tested against itself.
        """
        env = CartPoleEnv(b_cart=0.08, b_base=0.01, b_amplitude=0.02)
        env.reset(seed=0)
        p = Params.from_env(env)
        self.assertEqual((p.b_cart, p.b_base, p.b_amp), (0.08, 0.01, 0.02))

        for z in (Z0, np.zeros(4), np.array([0.2, np.pi, -0.3, 0.4])):
            with self.subTest(state=tuple(z)):
                np.testing.assert_allclose(
                    dyn.drift(z, U0, p), env._drift(z, U0),
                    rtol=RTOL_ENV, atol=ATOL_ENV)
                np.testing.assert_allclose(
                    np.array(dyn.damping_raw(z[1], p)),
                    np.array(env.compute_damping(z[1])),
                    rtol=RTOL_ENV, atol=ATOL_ENV)

        xi = -0.413

        class FixedRng:
            def standard_normal(self):
                return xi

        env.state = Z0.copy()
        env._np_random = FixedRng()
        theirs = np.asarray(env.integrator(np.array([U0]), DT))
        np.testing.assert_allclose(
            dyn.heun_step(Z0, U0, np.sqrt(DT) * xi, DT, p), theirs,
            rtol=RTOL_ENV, atol=ATOL_ENV)

    def test_damping_defaults_are_zero_so_existing_runs_are_unchanged(self):
        p = Params.from_env(CartPoleEnv())
        self.assertEqual((p.b_cart, p.b_base, p.b_amp), (0.0, 0.0, 0.0))

    def test_heun_step_at_zero_noise_is_deterministic_heun(self):
        p0 = replace(self.env_params, sigma_gust=0.0, sigma_turb=0.0)
        f0 = dyn.drift(Z0, U0, p0)
        expected = Z0 + 0.5 * (f0 + dyn.drift(Z0 + f0 * DT, U0, p0)) * DT
        # dW is deliberately nonzero: at sigma = 0 it must not matter.
        np.testing.assert_allclose(
            dyn.heun_step(Z0, U0, 1.234, DT, p0), expected, rtol=0, atol=1e-15)

    # -- the torch backend --------------------------------------------------
    #
    # The `_raw` functions are written against an array-module argument so that
    # one source serves numpy and torch.  Training runs the torch path, so this
    # agreement is load-bearing: if the two backends ever disagree, the model
    # being fitted is not the model that generated the data.

    def _torch_params(self):
        v = torch.tensor(self.params.to_vector(), dtype=torch.float64)
        return Params(*(v[i] for i in range(dyn.N_PARAM)),
                      smooth_eps=self.params.smooth_eps)

    def test_raw_functions_agree_on_torch_tensors(self):
        tp = self._torch_params()
        cases = {
            "generic": Z0,
            "upright at rest (thdot = 0, smoothed |.| non-smooth)": np.zeros(4),
            "pole horizontal (cos theta = 0)": np.array([0.0, np.pi / 2, 0.3, -0.2]),
            "hanging": np.array([-1.0, np.pi, 0.1, 0.1]),
        }
        for name, z_np in cases.items():
            z = torch.tensor(z_np, dtype=torch.float64)
            ut = torch.as_tensor(U0, dtype=torch.float64)
            with self.subTest(state=name):
                np.testing.assert_allclose(
                    torch.stack(dyn.drift_raw(z, ut, tp, xp=torch)).numpy(),
                    dyn.drift(z_np, U0, self.params), rtol=1e-13, atol=1e-15)
                np.testing.assert_allclose(
                    float(dyn.diffusion_raw(z, tp, xp=torch)),
                    dyn.diffusion(z_np, self.params), rtol=1e-13, atol=1e-15)
                np.testing.assert_allclose(
                    float(dyn.hamiltonian_raw(z, tp, xp=torch)),
                    dyn.hamiltonian(z_np, self.params), rtol=1e-13, atol=1e-15)
                np.testing.assert_allclose(
                    torch.stack(dyn.damping_raw(z[1], tp, xp=torch)).numpy(),
                    np.array(dyn.damping_raw(z_np[1], self.params)),
                    rtol=1e-13, atol=1e-15)

    def test_drift_is_differentiable_through_torch(self):
        """A gradient exists and flows to every parameter the drift depends on.

        This is what replaces the analytic Jacobians: autodiff is now the only
        route to a gradient, so it gets a test of its own.
        """
        v = torch.tensor(self.params.to_vector(), dtype=torch.float64,
                         requires_grad=True)
        tp = Params(*(v[i] for i in range(dyn.N_PARAM)),
                    smooth_eps=self.params.smooth_eps)
        z = torch.tensor(Z0, dtype=torch.float64)
        torch.stack(dyn.drift_raw(z, torch.as_tensor(U0, dtype=torch.float64),
                                  tp, xp=torch)).sum().backward()
        g = v.grad.numpy()
        self.assertTrue(np.all(np.isfinite(g)))
        for k in dyn.MECHANICAL + dyn.DAMPING:
            with self.subTest(param=PARAM_NAMES[k]):
                self.assertNotEqual(g[k], 0.0)
        # Wind is absent from the drift, so those gradients must be exactly zero.
        for k in dyn.WIND:
            with self.subTest(param=PARAM_NAMES[k]):
                self.assertEqual(g[k], 0.0)


if __name__ == "__main__":
    unittest.main()
