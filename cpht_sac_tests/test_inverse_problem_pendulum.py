# tests/test_inverse_problem_pendulum.py
"""
Agreement tests for the windy-pendulum forward model.

Same two claims as the cart-pole suite: that inverse_problem.pendulum is the
SAME MAP as environment/windypendulumenv.py, and that the port-Hamiltonian
assembly reproduces it.  The pendulum's env draws its noise with
`self._np_random.normal(0.0, 1)` rather than `standard_normal()`, so the stub
below differs from the cart-pole one -- a detail that silently breaks the
comparison if copied across.
"""

import unittest
from dataclasses import replace

import numpy as np
import torch

from environment.windypendulumenv import PendulumParams, WindyPendulumEnv
from inverse_problem import pendulum as pend
from inverse_problem.pendulum import Params

Z0 = np.array([0.7, 1.3])        # neither hanging nor upright, p nonzero
U0 = 3.5
DT = 0.02
RTOL, ATOL = 1e-13, 1e-14


def make_env(**kw):
    e = WindyPendulumEnv(params=PendulumParams(), dt=DT, max_steps=200, **kw)
    e.reset(seed=0)
    return e


class FixedRng:
    """WindyPendulumEnv.integrator calls .normal(0.0, 1), not .standard_normal()."""

    def __init__(self, xi):
        self.xi = xi

    def normal(self, *_a, **_k):
        return self.xi


class PendulumForwardModelTest(unittest.TestCase):
    def setUp(self):
        self.env = make_env()
        self.p = Params.from_env(self.env)

    # -- coordinates --------------------------------------------------------

    def test_combo_values_match_env(self):
        e = self.env
        self.assertAlmostEqual(self.p.inertia, e.params.m * e.params.L**2, places=15)
        self.assertAlmostEqual(self.p.inertia, e.I, places=15)
        self.assertAlmostEqual(self.p.gamma, e.params.m * e.params.g * e.params.L, places=15)
        self.assertAlmostEqual(self.p.b_base, e.params.b_base, places=15)
        self.assertAlmostEqual(self.p.b_amp, e.params.b_amplitude, places=15)

    def test_physical_roundtrip(self):
        phys = dict(m=1.3, L=0.7, g=9.81)
        back = Params.from_physical(**phys).to_physical(phys["g"])
        for k, v in phys.items():
            self.assertAlmostEqual(back[k], v, places=12, msg=k)

    def test_vector_roundtrip(self):
        v = self.p.to_vector()
        self.assertEqual(list(v), [getattr(self.p, n) for n in pend.PARAM_NAMES])
        self.assertEqual(Params.from_vector(v), self.p)

    # -- same map as the environment ----------------------------------------

    def test_drift_matches_env_force(self):
        for z in (Z0, np.array([0.0, 0.0]), np.array([np.pi, -2.0]), np.array([-2.5, 4.0])):
            with self.subTest(state=tuple(z)):
                thetadot, pdot = pend.drift(z, U0, self.p)
                self.assertAlmostEqual(thetadot, z[1] / self.env.I, delta=ATOL)
                self.assertAlmostEqual(pdot, self.env.force(z[0], z[1], U0), delta=ATOL)

    def test_damping_matches_env(self):
        for th in (0.0, 0.3, np.pi / 2, np.pi, -1.1):
            with self.subTest(theta=th):
                self.assertAlmostEqual(pend.damping_raw(th, self.p),
                                       self.env.compute_damping(th), places=15)

    def test_diffusion_matches_env(self):
        for z in (Z0, np.array([0.0, 0.0]), np.array([1.0, -5.0])):
            with self.subTest(state=tuple(z)):
                self.assertAlmostEqual(pend.diffusion(z, self.p),
                                       self.env.diffusion(z[1]), places=15)

    def test_heun_step_matches_env(self):
        xi = 0.6131
        self.env._np_random = FixedRng(xi)
        theirs = np.asarray(self.env.integrator(Z0[0], Z0[1], U0, DT))
        mine = pend.heun_step(Z0, U0, np.sqrt(DT) * xi, DT, self.p)
        np.testing.assert_allclose(mine, theirs, rtol=RTOL, atol=ATOL)

    def test_rollout_matches_env_over_many_steps(self):
        n = 40
        xis = np.random.default_rng(3).standard_normal(n)
        u = np.full(n, 2.0)

        class SeqRng:
            def __init__(self, seq):
                self.seq, self.i = seq, 0

            def normal(self, *_a, **_k):
                v = self.seq[self.i]
                self.i += 1
                return v

        self.env._np_random = SeqRng(xis)
        th, mom = Z0
        theirs = [np.array([th, mom])]
        for k in range(n):
            th, mom = self.env.integrator(th, mom, u[k], DT)
            theirs.append(np.array([th, mom]))

        mine, _, _ = pend.rollout(self.p, Z0, u, DT, dW=np.sqrt(DT) * xis)
        np.testing.assert_allclose(mine, np.array(theirs), rtol=1e-11, atol=1e-13)

    def test_zero_noise_step_ignores_dW(self):
        p0 = replace(self.p, sigma_gust=0.0, sigma_turb=0.0)
        a = pend.heun_step(Z0, U0, 0.0, DT, p0)
        b = pend.heun_step(Z0, U0, 7.5, DT, p0)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)

    # -- port-Hamiltonian structure -----------------------------------------

    def test_ph_assembly_equals_drift(self):
        rng = np.random.default_rng(1)
        for _ in range(300):
            z = rng.normal(size=2) * np.array([np.pi, 3.0])
            u = rng.uniform(-pend.MAX_TORQUE, pend.MAX_TORQUE)
            np.testing.assert_allclose(pend.drift_from_ph(z, u, self.p),
                                       pend.drift(z, u, self.p), rtol=1e-12, atol=1e-14)

    def test_J_skew_and_R_momentum_block(self):
        np.testing.assert_allclose(pend.J_PH, -pend.J_PH.T, atol=0)
        R = pend.dissipation_matrix(Z0, self.p)
        np.testing.assert_allclose(R, R.T, atol=0)
        self.assertTrue(np.all(np.linalg.eigvalsh(R) >= -1e-15))
        self.assertEqual(R[0, 0], 0.0)
        self.assertEqual(R[0, 1], 0.0)

    def test_grad_hamiltonian_vs_torch_autograd(self):
        p = self.p
        rng = np.random.default_rng(4)
        for _ in range(20):
            z_np = rng.normal(size=2) * np.array([np.pi, 3.0])
            z = torch.tensor(z_np, dtype=torch.float64, requires_grad=True)
            pend.hamiltonian_raw(z, p, xp=torch).backward()
            np.testing.assert_allclose(pend.grad_hamiltonian(z_np, p), z.grad.numpy(),
                                       rtol=1e-11, atol=1e-13)

    def test_potential_is_minimal_hanging_and_maximal_upright(self):
        """theta = 0 hangs, theta = pi is the control target.  Opposite of the
        cart-pole, and inverting it would silently invert the controller."""
        V = lambda th: pend.hamiltonian(np.array([th, 0.0]), self.p)
        self.assertLess(V(pend.HANGING), V(0.3))
        self.assertGreater(V(pend.UPRIGHT), V(np.pi - 0.3))
        self.assertAlmostEqual(V(pend.HANGING), -self.p.gamma, places=12)
        self.assertAlmostEqual(V(pend.UPRIGHT), self.p.gamma, places=12)

    def test_lossless_limit_conserves_energy(self):
        p0 = replace(self.p, b_base=0.0, b_amp=0.0, sigma_gust=0.0, sigma_turb=0.0)
        z, _, _ = pend.rollout(p0, np.array([0.4, 0.0]), np.zeros(1000), DT,
                               dW=np.zeros(1000))
        H = np.array([pend.hamiltonian(zk, p0) for zk in z])
        self.assertLess(np.max(np.abs(H - H[0])) / abs(H[0]), 1e-2)

    def test_damping_removes_energy(self):
        p0 = replace(self.p, sigma_gust=0.0, sigma_turb=0.0)
        z, _, _ = pend.rollout(p0, np.array([0.4, 0.0]), np.zeros(1000), DT,
                               dW=np.zeros(1000))
        H = np.array([pend.hamiltonian(zk, p0) for zk in z])
        self.assertLess(H[-1], H[0])
        self.assertLess(np.max(np.diff(H)), 1e-9)

    def test_damping_is_much_stronger_than_on_the_cartpole(self):
        """The motivation for switching plants, as a number.

        Damping torque against gravity torque at a representative state.  On the
        cart-pole this ratio was ~3.6% and D came back with 12% error; here it
        should be an order of magnitude larger.
        """
        z = np.array([np.pi / 2, 2.0])
        damp = pend.damping_raw(z[0], self.p) * z[1] / self.p.inertia
        grav = self.p.gamma * abs(np.sin(z[0]))
        self.assertGreater(abs(damp) / grav, 0.10)

    # -- data ---------------------------------------------------------------

    def test_rollout_reproducible_from_stored_dW(self):
        tr = pend.generate(self.p, n_steps=200, seed=5)
        again, _, _ = pend.rollout(self.p, tr.z[0], tr.u, tr.dt, dW=tr.dW)
        np.testing.assert_allclose(again, tr.z, atol=0)

    def test_torque_is_clipped(self):
        z, u, _ = pend.rollout(self.p, Z0, np.full(10, 500.0), DT,
                               rng=np.random.default_rng(0))
        self.assertTrue(np.all(u == pend.MAX_TORQUE))

    def test_rollout_rejects_bad_noise_arguments(self):
        with self.assertRaises(ValueError):
            pend.rollout(self.p, Z0, np.zeros(5), DT)
        with self.assertRaises(ValueError):
            pend.rollout(self.p, Z0, np.zeros(5), DT, dW=np.zeros(4))


if __name__ == "__main__":
    unittest.main()
