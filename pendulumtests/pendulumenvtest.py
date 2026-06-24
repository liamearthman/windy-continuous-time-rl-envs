from stable_baselines3.common.env_checker import check_env
from environment.PendulumEnv import PendulumEnv, PendulumParams



# env = PendulumEnv(params=PendulumParams())
# import numpy as np
# returns = []
# for _ in range(50):
#     obs, _ = env.reset()
#     ep_ret, done = 0.0, False
#     while not done:
#         obs, r, term, trunc, _ = env.step(env.action_space.sample())
#         ep_ret += r
#         done = term or trunc
#     returns.append(ep_ret)
# print("random return:", np.mean(returns), "+/-", np.std(returns))


from stable_baselines3 import SAC
env = PendulumEnv(params=PendulumParams())
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20_000)

import numpy as np
obs, _ = env.reset()
thetas, omegas = [], []
for _ in range(env.max_steps):
    a, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, _ = env.step(a)
    thetas.append(float(obs[0]))
    omegas.append(float(obs[1]) / env.I)   # angular velocity
    if term or trunc:
        break

import matplotlib.pyplot as plt
plt.plot(thetas, label="theta"); plt.axhline(np.pi, ls="--", label="goal pi"); plt.axhline(-np.pi, ls="--")
plt.legend(); plt.show()
print("final theta:", thetas[-1], "  goal: +/- pi")