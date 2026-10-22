# cpht_sac_tests/test_residual_env.py
"""
Tests for the residual-RL wrapper.

The claims worth pinning down are the ones that make the experiment mean
anything: that a zero nudge reproduces the base law exactly (so `max_nudge=0` is
a genuine no-RL baseline on the same code path), that the nudge is bounded (so
the policy corrects rather than replaces), and that the agent's action space is
the NUDGE rather than the torque (so models/actor.py rescales to the right box).
"""

import unittest

import numpy as np
from gymnasium import spaces

from environment.windypendulumenv import PendulumParams, WindyPendulumEnv
from inverse_problem import control as C
from inverse_problem.pendulum import Params
from inverse_problem.residual_env import ResidualIDAPBCPendulum, make_residual_env

KP, KD, NUDGE = 6.0, 2.5, 5.0


def true_params():
    e = WindyPendulumEnv(params=PendulumParams(), dt=0.02, max_steps=200)
    e.reset(seed=0)
    return Params.from_env(e)


class ResidualEnvTest(unittest.TestCase):
    def setUp(self):
        self.p = true_params()
        self.env = make_residual_env(self.p, max_nudge=NUDGE, K_p=KP, k_d=KD,
                                     max_steps=200, seed=0)

    def test_action_space_is_the_nudge_not_the_torque(self):
        """models/actor.py rescales a tanh output using action_space.low/high, so
        this box IS the nudge budget -- the agent cannot exceed it."""
        self.assertIsInstance(self.env.action_space, spaces.Box)
        self.assertEqual(self.env.action_space.shape, (1,))
        self.assertAlmostEqual(float(self.env.action_space.high[0]), NUDGE, places=5)
        self.assertAlmostEqual(float(self.env.action_space.low[0]), -NUDGE, places=5)
        # ... and it is NOT the parent's torque box
        self.assertLess(float(self.env.action_space.high[0]), self.env.max_torque)

    def test_observation_space_is_inherited_unchanged(self):
        plain = WindyPendulumEnv(params=PendulumParams(), dt=0.02,
                                 max_steps=200, q_only=False)
        self.assertEqual(self.env.observation_space.shape, plain.observation_space.shape)

    def test_zero_nudge_applies_exactly_the_base_law(self):
        self.env.reset(seed=1)
        for _ in range(8):
            expected = self.env.base_action()
            _, _, _, _, info = self.env.step(np.zeros(1, dtype=np.float32))
            self.assertAlmostEqual(info["u_base"], expected, places=12)
            self.assertAlmostEqual(info["nudge"], 0.0, places=12)
            self.assertAlmostEqual(info["u_total"], expected, places=12)

    def test_base_matches_the_standalone_controller(self):
        """The wrapper must not quietly reimplement the law."""
        env = make_residual_env(self.p, max_nudge=0.0, K_p=KP, k_d=KD,
                                max_steps=200, seed=3)
        env.reset(seed=3)
        ida = C.IDAPBC(self.p, K_p=KP, k_d=KD)
        for _ in range(6):
            z = np.array([env._theta, env._p])
            self.assertAlmostEqual(env.base_action(), ida.act(z, clip=False), places=12)
            env.step(np.zeros(1, dtype=np.float32))

    def test_nudge_is_clipped_to_the_budget(self):
        for req, want in [(2.0, 2.0), (1e6, NUDGE), (-1e6, -NUDGE)]:
            self.env.reset(seed=2)
            _, _, _, _, info = self.env.step(np.array([req], dtype=np.float32))
            self.assertAlmostEqual(info["nudge"], want, places=10)

    def test_total_torque_respects_the_plant_limit(self):
        """Nudge budget and torque limit are separate constraints; both hold."""
        env = make_residual_env(self.p, max_nudge=100.0, K_p=KP, k_d=KD,
                                max_steps=200, seed=4)
        env.reset(seed=4)
        _, _, _, _, info = env.step(np.array([100.0], dtype=np.float32))
        self.assertLessEqual(abs(info["u_total"]), env.max_torque + 1e-9)

    def test_max_nudge_zero_is_a_genuine_no_rl_baseline(self):
        """Whatever the agent proposes is ignored, so the baseline cannot be
        contaminated by a stray action."""
        env = make_residual_env(self.p, max_nudge=0.0, K_p=KP, k_d=KD,
                                max_steps=200, seed=5)
        env.reset(seed=5)
        base = env.base_action()
        _, _, _, _, info = env.step(np.array([7.3], dtype=np.float32))
        self.assertAlmostEqual(info["nudge"], 0.0, places=12)
        self.assertAlmostEqual(info["u_total"], base, places=12)

    def test_info_carries_the_decomposition(self):
        self.env.reset(seed=6)
        _, _, _, _, info = self.env.step(np.array([1.5], dtype=np.float32))
        for k in ("u_base", "nudge", "u_total"):
            self.assertIn(k, info)
        self.assertAlmostEqual(info["u_base"] + info["nudge"], info["u_total"], places=10)

    def test_learned_and_true_base_params_give_different_torques(self):
        """Guards the ablation: --oracle-base must actually change something."""
        wrong = Params(inertia=self.p.inertia * 1.4, gamma=self.p.gamma * 0.7,
                       b_base=0.0, b_amp=0.0,
                       sigma_gust=self.p.sigma_gust, sigma_turb=self.p.sigma_turb)
        a = make_residual_env(self.p, max_nudge=0.0, K_p=KP, k_d=KD, max_steps=200, seed=7)
        b = make_residual_env(wrong, max_nudge=0.0, K_p=KP, k_d=KD, max_steps=200, seed=7)
        a.reset(seed=7); b.reset(seed=7)
        b._theta, b._p = a._theta, a._p
        self.assertNotAlmostEqual(a.base_action(), b.base_action(), places=3)

    def test_episode_runs_to_completion_under_a_random_policy(self):
        env = make_residual_env(self.p, max_nudge=NUDGE, K_p=KP, k_d=KD,
                                max_steps=100, seed=8)
        obs, _ = env.reset(seed=8)
        rng = np.random.default_rng(0)
        for _ in range(100):
            a = rng.uniform(-NUDGE, NUDGE, size=1).astype(np.float32)
            obs, r, term, trunc, info = env.step(a)
            self.assertTrue(np.all(np.isfinite(obs)))
            self.assertTrue(np.isfinite(r))
            if term or trunc:
                break

    def test_subclass_relationship(self):
        """It is the same env, not a parallel reimplementation."""
        self.assertIsInstance(self.env, WindyPendulumEnv)
        self.assertIsInstance(self.env, ResidualIDAPBCPendulum)


if __name__ == "__main__":
    unittest.main()
