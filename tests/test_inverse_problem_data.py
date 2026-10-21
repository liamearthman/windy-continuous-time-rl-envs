# tests/test_inverse_problem_data.py
"""
Tests for synthetic data generation and for the port-Hamiltonian structure.

The pH tests are the substantive ones.  They assert that the windy cart-pole is
canonically Hamiltonian with ZERO dissipation -- not as a modelling choice but
as a fact about environment/cartpole.py:

    J grad H(z) + G_INPUT * u  ==  drift(z, u)      exactly, R = 0

and that the simulated DATA carries that structure, via the Stratonovich energy
balance dH = u xdot dt + sigma thetadot dW.  The second claim needs a
convergence study rather than a single tolerance, because the balance is exact
only in the dt -> 0 limit; at finite dt the residual is the integrator's error.
"""

import unittest
from dataclasses import replace

import numpy as np
import torch

from environment.cartpole import CartPoleEnv
from inverse_problem import data, dynamics as dyn
from inverse_problem.dynamics import Params

Z0 = np.array([0.0, 0.05, 0.0, 0.0])


class PortHamiltonianStructureTest(unittest.TestCase):
    def setUp(self):
        env = CartPoleEnv()
        env.reset(seed=0)
        self.p = Params.from_env(env)

    def test_J_is_skew_symplectic(self):
        J = dyn.J_PH
        np.testing.assert_allclose(J, -J.T, atol=0)
        np.testing.assert_allclose(J @ J, -np.eye(4), atol=0)

    def test_ph_assembly_equals_drift(self):
        """J grad H + G u == drift, over random states including large angles."""
        rng = np.random.default_rng(1)
        for _ in range(200):
            z = rng.normal(size=4) * np.array([1.0, np.pi, 1.0, 0.5])
            u = rng.uniform(-data.MAX_FORCE, data.MAX_FORCE)
            np.testing.assert_allclose(
                dyn.drift_from_ph(z, u, self.p),
                dyn.drift(z, u, self.p), rtol=1e-12, atol=1e-14)

    def test_no_dissipation_at_zero_damping(self):
        """At R = 0 the lossless assembly alone reproduces the drift.

        CartPoleEnv's damping defaults to zero, so this is the DEFAULT
        configuration, not a property of the plant -- the cart-pole gained a real
        R when b_cart/b_base/b_amplitude were added.  What this pins down is that
        turning damping off recovers the frictionless system exactly, which is
        what makes the zero-damping run a usable integrator-error baseline.
        """
        p0 = replace(self.p, b_cart=0.0, b_base=0.0, b_amp=0.0)
        rng = np.random.default_rng(2)
        for _ in range(100):
            z = rng.normal(size=4) * np.array([1.0, np.pi, 1.0, 0.5])
            residual = dyn.drift(z, 0.0, p0) - dyn.J_PH @ dyn.grad_hamiltonian(z, p0)
            np.testing.assert_allclose(residual, np.zeros(4), atol=1e-14)

    def test_ph_assembly_with_damping(self):
        """(J - R) grad H + G u == drift, with damping on."""
        pd = replace(self.p, b_cart=0.08, b_base=0.01, b_amp=0.02)
        rng = np.random.default_rng(5)
        for _ in range(200):
            z = rng.normal(size=4) * np.array([1.0, np.pi, 1.0, 0.5])
            u = rng.uniform(-data.MAX_FORCE, data.MAX_FORCE)
            np.testing.assert_allclose(
                dyn.drift_from_ph(z, u, pd), dyn.drift(z, u, pd),
                rtol=1e-12, atol=1e-14)

    def test_dissipation_matrix_is_psd_and_momentum_block_only(self):
        pd = replace(self.p, b_cart=0.08, b_base=0.01, b_amp=0.02)
        rng = np.random.default_rng(6)
        for _ in range(50):
            z = rng.normal(size=4) * np.array([1.0, np.pi, 1.0, 0.5])
            R = dyn.dissipation_matrix(z, pd)
            np.testing.assert_allclose(R, R.T, atol=0)
            self.assertTrue(np.all(np.linalg.eigvalsh(R) >= -1e-15))
            # Position block must be untouched: Rayleigh damping acts on
            # velocities, so R lives entirely in the momentum rows/columns.
            np.testing.assert_allclose(R[:2, :], np.zeros((2, 4)), atol=0)
            np.testing.assert_allclose(R[:, :2], np.zeros((4, 2)), atol=0)

    def test_damping_removes_energy_monotonically(self):
        """With u = 0 and sigma = 0, Hdot = -grad H^T R grad H <= 0.

        The physics claim behind a D-network: damping must take energy OUT, and
        at a rate set by R.  This also establishes that the damping defaults are
        large enough to be seen above the integrator's own drift -- the decay
        here must exceed the zero-damping floor by a wide margin, or there is
        nothing for an estimator to find.
        """
        shut = dict(sigma_gust=0.0, sigma_turb=0.0)
        p_damped = replace(self.p, b_cart=0.08, b_base=0.01, b_amp=0.02, **shut)
        p_free = replace(self.p, b_cart=0.0, b_base=0.0, b_amp=0.0, **shut)

        z0 = data.initial_state(p_damped, theta=0.6, thetadot=1.5, xdot=0.5)
        kw = dict(n_steps=1000, dt=0.02, z0=z0, input_design="zero", seed=0)
        damped = data.generate(p_damped, **kw)
        free = data.generate(p_free, **kw)

        H_d = data.hamiltonian_series(damped)
        H_f = data.hamiltonian_series(free)

        # Monotone non-increasing, up to the integrator's own wobble.
        self.assertLess(np.max(np.diff(H_d)), 1e-6)
        # And the real decay dwarfs the lossless run's drift.
        decay = H_d[0] - H_d[-1]
        floor = np.max(np.abs(H_f - H_f[0]))
        self.assertGreater(decay, 0.0)
        self.assertGreater(decay, 100 * floor,
                           f"damping decay {decay:.4g} not clear of integrator "
                           f"floor {floor:.4g} -- defaults are too small to identify")
        self.assertFalse(data.energy_drift(damped)["interpretable"])
        self.assertTrue(data.energy_drift(free)["interpretable"])

    def test_lossless_power_identity(self):
        """grad H^T J grad H == 0, so Hdot = u xdot with no input."""
        rng = np.random.default_rng(3)
        for _ in range(100):
            z = rng.normal(size=4) * np.array([1.0, np.pi, 1.0, 0.5])
            g = dyn.grad_hamiltonian(z, self.p)
            self.assertAlmostEqual(float(g @ dyn.J_PH @ g), 0.0, delta=1e-12)

    def test_grad_hamiltonian_vs_torch_autograd(self):
        """The closed form of grad H against autodiff of H itself."""
        rng = np.random.default_rng(4)
        v = torch.tensor(self.p.to_vector(), dtype=torch.float64)
        tp = Params(*(v[i] for i in range(dyn.N_PARAM)), smooth_eps=self.p.smooth_eps)
        for _ in range(20):
            z_np = rng.normal(size=4) * np.array([1.0, np.pi, 1.0, 0.5])
            z = torch.tensor(z_np, dtype=torch.float64, requires_grad=True)
            dyn.hamiltonian_raw(z, tp, xp=torch).backward()
            np.testing.assert_allclose(
                dyn.grad_hamiltonian(z_np, self.p), z.grad.numpy(),
                rtol=1e-10, atol=1e-13)

    def test_momentum_half_of_grad_H_is_the_velocities(self):
        z = np.array([0.3, 0.4, -0.7, 0.25])
        np.testing.assert_allclose(
            dyn.grad_hamiltonian(z, self.p)[2:], dyn.velocities(z, self.p), atol=1e-15)

    # -- the balance in the simulated data ----------------------------------

    def _residual_at(self, p, dt, dW, u):
        z, u, dW = data.rollout(p, Z0, u, dt, dW=dW)
        tr = data.Trajectory(t=np.arange(len(z)) * dt, z=z, u=u, dW=dW, obs=z,
                             dt=dt, params=p, sensing="full",
                             obs_noise_std=np.zeros(4), seed=None)
        return np.max(np.abs(data.port_power_residual(tr)))

    def test_stratonovich_port_balance_converges(self):
        """dH - u xdot dt - sigma thetadot dW -> 0 as dt -> 0.

        ONE Brownian path, coarsened by summing increments, and an input that is
        a function of TIME not of step index -- otherwise each dt is a different
        trajectory and the comparison is meaningless.
        """
        dt_fine, T = 0.0025, 2.0
        rng = np.random.default_rng(7)
        dW_fine = np.sqrt(dt_fine) * rng.standard_normal(int(T / dt_fine))

        for label, p in [
            ("lossless", replace(self.p, sigma_gust=0.0, sigma_turb=0.0)),
            ("windy", replace(self.p, sigma_gust=0.05, sigma_turb=0.02)),
        ]:
            res = []
            for m in (8, 4, 2, 1):
                dt = dt_fine * m
                dW = dW_fine.reshape(-1, m).sum(axis=1)
                u = 5.0 * np.sin(2 * np.pi * 0.7 * np.arange(len(dW)) * dt)
                res.append(self._residual_at(p, dt, dW, u))
            with self.subTest(case=label):
                self.assertTrue(all(b < a for a, b in zip(res, res[1:])),
                                f"residual not monotone decreasing: {res}")
                # Across an 8x refinement, 8x would be first order and 64x
                # second.  The lossless case comes in near 512x (third order --
                # the trapezoidal port rule is exact to one extra power when
                # there is no noise).  The windy case lands near 48x, i.e.
                # overall order ~1.9 and degrading at the fine end: stochastic
                # Heun with STATE-DEPENDENT diffusion has strong order below 2,
                # so the stochastic port term caps the rate and no amount of
                # refinement recovers second order.  Requiring clearly
                # superlinear convergence is the honest assertion -- it
                # establishes the balance is real rather than fitted, without
                # claiming an order the scheme does not have.
                floor = 64.0 if label == "lossless" else 16.0
                self.assertGreater(res[0] / res[-1], floor,
                                   f"convergence too slow for {label}: {res}")

    def test_energy_conserved_when_both_ports_shut(self):
        p0 = replace(self.p, sigma_gust=0.0, sigma_turb=0.0)
        tr = data.generate(p0, n_steps=500, dt=0.02, input_design="zero", seed=0)
        d = data.energy_drift(tr)
        self.assertTrue(d["interpretable"])
        # Heun is not symplectic, so this is small-but-nonzero by design.
        self.assertLess(d["max_dev_rel"], 1e-2)

    def test_energy_drift_flagged_uninterpretable_when_ports_open(self):
        tr = data.generate(self.p, n_steps=50, dt=0.02, input_design="sinusoid", seed=0)
        self.assertFalse(data.energy_drift(tr)["interpretable"])


class SyntheticDataTest(unittest.TestCase):
    def setUp(self):
        self.env = CartPoleEnv()
        self.env.reset(seed=0)
        self.p = Params.from_env(self.env)

    def test_rollout_matches_env_over_many_steps(self):
        """The whole rollout, not one step, against the env's own integrator."""
        n = 50
        rng = np.random.default_rng(11)
        xis = rng.standard_normal(n)
        u = np.full(n, 3.0)
        dt = 0.02

        class SeqRng:
            def __init__(self, seq):
                self.seq, self.i = seq, 0

            def standard_normal(self):
                v = self.seq[self.i]
                self.i += 1
                return v

        self.env.state = Z0.copy()
        self.env._np_random = SeqRng(xis)
        theirs = [Z0.copy()]
        for k in range(n):
            theirs.append(np.asarray(self.env.integrator(np.array([u[k]]), dt)))
            self.env.state = theirs[-1]

        mine, _, _ = data.rollout(self.p, Z0, u, dt, dW=np.sqrt(dt) * xis)
        np.testing.assert_allclose(mine, np.array(theirs), rtol=1e-11, atol=1e-13)

    def test_rollout_is_reproducible_from_stored_dW(self):
        tr = data.generate(self.p, n_steps=100, seed=5)
        again, _, _ = data.rollout(self.p, tr.z[0], tr.u, tr.dt, dW=tr.dW)
        np.testing.assert_allclose(again, tr.z, atol=0)

    def test_rollout_rejects_mismatched_dW(self):
        with self.assertRaises(ValueError):
            data.rollout(self.p, Z0, np.zeros(10), 0.02, dW=np.zeros(9))
        with self.assertRaises(ValueError):
            data.rollout(self.p, Z0, np.zeros(10), 0.02)

    def test_input_is_clipped_like_the_env(self):
        tr = data.generate(self.p, n_steps=10, input_design=np.full(10, 50.0))
        self.assertTrue(np.all(tr.u == data.MAX_FORCE))

    def test_initial_state_roundtrip(self):
        z0 = data.initial_state(self.p, x=0.2, theta=0.3, xdot=-0.4, thetadot=1.1)
        xdot, thdot = dyn.velocities_raw(z0[1], z0[2], z0[3], self.p)
        self.assertAlmostEqual(xdot, -0.4, places=13)
        self.assertAlmostEqual(thdot, 1.1, places=13)

    def test_sensing_modes(self):
        full = data.generate(self.p, n_steps=20, sensing="full")
        pos = data.generate(self.p, n_steps=20, sensing="positions")
        self.assertEqual(full.obs.shape, (21, 4))
        self.assertEqual(pos.obs.shape, (21, 2))
        np.testing.assert_allclose(pos.obs, pos.z[:, :2], atol=0)
        with self.assertRaises(ValueError):
            data.generate(self.p, n_steps=5, sensing="velocity")

    def test_observation_noise_is_per_channel(self):
        rng = np.random.default_rng(0)
        z = np.zeros((20000, 4))
        obs, std = data.observe(z, "full", noise_std=[0.1, 0.01, 0.0, 0.0], rng=rng)
        np.testing.assert_allclose(std, [0.1, 0.01, 0.0, 0.0])
        np.testing.assert_allclose(obs.std(axis=0), [0.1, 0.01, 0.0, 0.0], rtol=0.05)
        self.assertTrue(np.all(obs[:, 2:] == 0.0))

    def test_observe_needs_rng_for_nonzero_noise(self):
        with self.assertRaises(ValueError):
            data.observe(np.zeros((3, 4)), "full", noise_std=0.1)
        with self.assertRaises(ValueError):
            data.observe(np.zeros((3, 4)), "full", noise_std=-0.1, rng=np.random.default_rng(0))

    def test_true_obs_is_noise_free(self):
        tr = data.generate(self.p, n_steps=20, sensing="positions",
                           obs_noise_std=0.05, seed=1)
        np.testing.assert_allclose(tr.true_obs(), tr.z[:, :2], atol=0)
        self.assertGreater(np.abs(tr.obs - tr.true_obs()).max(), 0.0)

    def test_random_step_hold_is_dt_invariant_in_seconds(self):
        """A 0.2 s hold must stay 0.2 s of signal when dt changes."""
        a = data.input_random_steps(100, 0.02, np.random.default_rng(0), hold_seconds=0.2)
        b = data.input_random_steps(200, 0.01, np.random.default_rng(0), hold_seconds=0.2)
        np.testing.assert_allclose(a, b[::2], atol=0)

    def test_save_load_roundtrip(self):
        import tempfile, os
        tr = data.generate(self.p, n_steps=30, sensing="positions",
                           obs_noise_std=0.01, seed=3)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "traj.npz")
            tr.save(path)
            back = data.Trajectory.load(path)
        for f in ("t", "z", "u", "dW", "obs", "obs_noise_std"):
            np.testing.assert_allclose(getattr(back, f), getattr(tr, f), atol=0)
        self.assertEqual(back.params, tr.params)
        self.assertEqual((back.dt, back.sensing, back.seed), (tr.dt, tr.sensing, tr.seed))

    def test_free_swing_leaves_the_rl_valid_region(self):
        """Documents the 'no termination' choice: this data is not RL-valid."""
        tr = data.generate(self.p, n_steps=500, dt=0.02, input_design="zero", seed=0)
        self.assertGreater(np.abs(tr.z[:, 1]).max(), np.deg2rad(12))


if __name__ == "__main__":
    unittest.main()
