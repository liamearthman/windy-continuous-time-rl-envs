# tests/test_inverse_problem_control.py
"""
Tests for the IDA-PBC law and the closed-loop harness.

The substantive ones check the Lyapunov claims the derivation rests on -- that
H_d has its minimum at upright, that it is non-increasing along deterministic
closed-loop trajectories, and that hanging is repelling rather than attracting.
If any of those fail the control law is wrong regardless of how the simulation
happens to look.
"""

import unittest
from dataclasses import replace

import numpy as np

from environment.windypendulumenv import PendulumParams, WindyPendulumEnv
from inverse_problem import control as C
from inverse_problem import pendulum as pend
from inverse_problem.pendulum import Params


def true_params():
    e = WindyPendulumEnv(params=PendulumParams(), dt=0.02, max_steps=200)
    e.reset(seed=0)
    return Params.from_env(e)


class IDAPBCTest(unittest.TestCase):
    def setUp(self):
        self.p = true_params()
        self.p_det = replace(self.p, sigma_gust=0.0, sigma_turb=0.0)
        self.ida = C.IDAPBC(self.p, K_p=6.0, k_d=2.5)

    def test_wrap_to_pi(self):
        # Range is [-pi, pi): exactly pi folds to -pi, the same point.
        for a, b in [(0.0, 0.0), (np.pi, -np.pi), (3 * np.pi, -np.pi),
                     (-3 * np.pi, -np.pi), (2 * np.pi + 0.3, 0.3), (-0.4, -0.4)]:
            self.assertAlmostEqual(float(C.wrap_to_pi(a)), b, places=12)
        # and the invariant callers actually rely on
        for a in np.linspace(-20, 20, 97):
            self.assertAlmostEqual(np.sin(C.wrap_to_pi(a)), np.sin(a), places=12)

    def test_Hd_is_minimal_at_the_target(self):
        th = np.linspace(-np.pi, np.pi, 721)
        H = np.array([self.ida.energy([t, 0.0]) for t in th])
        # +pi and -pi are the same point, so compare on the circle.
        self.assertAlmostEqual(abs(float(C.wrap_to_pi(th[np.argmin(H)] - np.pi))), 0.0, delta=0.02)
        self.assertAlmostEqual(th[np.argmax(H)], 0.0, delta=0.02)
        self.assertLess(self.ida.energy([np.pi, 0.0]), self.ida.energy([0.0, 0.0]))

    def test_control_vanishes_at_both_equilibria(self):
        """Upright and hanging are both closed-loop equilibria; the law must not
        fight at either.  What distinguishes them is stability, not the torque."""
        for th in (np.pi, 0.0):
            self.assertAlmostEqual(self.ida.act([th, 0.0]), 0.0, delta=1e-12)

    def test_control_pushes_toward_upright(self):
        """Just below upright the torque must raise theta, and just above lower it."""
        self.assertGreater(self.ida.act([np.pi - 0.2, 0.0]), 0.0)
        self.assertLess(self.ida.act([np.pi + 0.2, 0.0]), 0.0)
        # ... and the opposite sense about hanging, which is now repelling.
        self.assertGreater(self.ida.act([0.2, 0.0]), 0.0)
        self.assertLess(self.ida.act([-0.2, 0.0]), 0.0)

    def test_Hd_is_non_increasing_without_wind(self):
        """Hdot_d = -k_d thetadot^2 <= 0 is the whole stability argument."""
        r = C.run_closed_loop(self.ida.act, self.p_det, [0.6, 1.0], n_steps=900, seed=0)
        H = np.array([self.ida.energy(z) for z in r.z])
        self.assertLess(np.max(np.diff(H)), 1e-6)
        self.assertLess(H[-1], H[0])

    def test_damping_cancellation_changes_the_torque(self):
        """cancel_damping is a real term, not a flag nobody reads."""
        on = C.IDAPBC(self.p, K_p=6.0, k_d=2.5, cancel_damping=True)
        off = C.IDAPBC(self.p, K_p=6.0, k_d=2.5, cancel_damping=False)
        z = [1.0, 2.0]
        self.assertNotAlmostEqual(on.act(z), off.act(z), places=3)
        expected = pend.damping_raw(z[0], self.p) * z[1] / self.p.inertia
        self.assertAlmostEqual(on.act(z) - off.act(z), expected, places=10)

    def test_stabilises_from_several_starts_without_wind(self):
        for z0 in ([np.pi - 0.3, 0.0], [1.0, 4.0], [-0.5, -1.0]):
            with self.subTest(z0=z0):
                r = C.run_closed_loop(self.ida.act, self.p_det, z0, n_steps=1500, seed=0)
                self.assertIsNotNone(r.settled_at, f"never settled from {z0}")
                self.assertLess(r.final_err, 0.02)

    def test_hanging_at_rest_is_a_saddle_the_wind_escapes(self):
        """From exactly (0, 0) the deterministic law produces no torque and the
        pendulum stays -- correct, it is the repelling maximum of V_d and a
        measure-zero start.  With wind the noise knocks it off and it recovers,
        which is worth pinning down because it looks like a bug otherwise."""
        still = C.run_closed_loop(self.ida.act, self.p_det, [0.0, 0.0], n_steps=600, seed=0)
        self.assertIsNone(still.settled_at)
        windy = C.run_closed_loop(self.ida.act, self.p, [0.0, 0.0], n_steps=1500, seed=0)
        self.assertIsNotNone(windy.settled_at)

    def test_torque_budget_and_feasible_Kp(self):
        kp = C.feasible_Kp(self.p)
        self.assertLess(self.p.gamma + kp, pend.MAX_TORQUE)
        r = C.run_closed_loop(self.ida.act, self.p, [1.0, 2.0], n_steps=900, seed=1)
        self.assertLess(r.saturation, 0.05)

    def test_act_respects_the_torque_limit(self):
        hot = C.IDAPBC(self.p, K_p=50.0, k_d=30.0)
        self.assertLessEqual(abs(hot.act([1.2, 9.0])), pend.MAX_TORQUE + 1e-9)
        self.assertGreater(abs(hot.act([1.2, 9.0], clip=False)), pend.MAX_TORQUE)

    def test_wind_leaves_a_steady_state_error_floor(self):
        """The thing the RL nudge exists to attack: with wind the controller
        cannot hold upright exactly, and the residual is far above the
        deterministic case."""
        det = C.run_closed_loop(self.ida.act, self.p_det, [np.pi - 0.3, 0.0], n_steps=1500, seed=0)
        wind = C.run_closed_loop(self.ida.act, self.p, [np.pi - 0.3, 0.0], n_steps=1500, seed=0)
        self.assertLess(det.rms_err_last, 1e-3)
        self.assertGreater(wind.rms_err_last, 10 * det.rms_err_last)


class ResidualPolicyTest(unittest.TestCase):
    def setUp(self):
        self.p = true_params()
        self.base = C.IDAPBC(self.p, K_p=6.0, k_d=2.5)

    def test_zero_nudge_reproduces_the_base_law(self):
        pi = C.residual_policy(self.base, lambda z: 0.0)
        for z in ([1.0, 2.0], [np.pi - 0.1, -0.5], [0.0, 0.0]):
            self.assertAlmostEqual(pi(z), self.base.act(z, clip=False), places=12)

    def test_nudge_is_bounded(self):
        pi = C.residual_policy(self.base, lambda z: 1e6, max_nudge=5.0)
        z = [1.0, 2.0]
        self.assertAlmostEqual(pi(z) - self.base.act(z, clip=False), 5.0, places=10)

    def test_a_bounded_nudge_cannot_erase_the_base_law(self):
        """The design constraint that keeps this from being plain model-free RL."""
        adversary = C.residual_policy(self.base, lambda z: -1e6, max_nudge=5.0)
        z = [np.pi - 0.5, 0.0]
        self.assertGreater(self.base.act(z, clip=False) - adversary(z), 0.0)
        self.assertAlmostEqual(self.base.act(z, clip=False) - adversary(z), 5.0, places=10)


if __name__ == "__main__":
    unittest.main()
