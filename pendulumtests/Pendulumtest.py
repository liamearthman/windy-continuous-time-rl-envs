import numpy as np

from environment.PendulumEnv import PendulumEnv, PendulumParams
env = PendulumEnv(params=PendulumParams())

# obs, info = env.reset(seed=0)
# print("reset obs:", obs, type(obs), getattr(obs, "shape", None))
# print("reset info:", info)

# for t in range(105):                      # past 100 so you can see truncation fire
#     action = env.action_space.sample()    # random action in your Box(±2)
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(t, "obs:", obs, "r:", reward, "action:", action, "term:", terminated, "trunc:", truncated)
#     if terminated or truncated:
#         print("episode ended at step", t)
#         break





### this is just to test if the Hamiltonian remains conservative 

I, m, L, g = 1.0, 1.0, 1.0, 9.81
dt = 0.02
theta, p = 1, 0  # small swing, no damping, no torque

def H(th, p): return p**2/(2*I) + m*g*L*(1 - np.cos(th))

# temporarily run with Ib_base=0, b_amplitude=0 so D=0
env = PendulumEnv(dt=dt, params=PendulumParams())
env.b_base = 0.3; env.b_amplitude = 0.3   # kill damping for the test

Hs = []
for _ in range(2000):
    # print(env.reset())
    theta, p = env.integrator(theta, p, u=0.0, dt=dt)
    Hs.append(H(theta, p))

print(Hs[0])
print(min(Hs), max(Hs), max(Hs)-min(Hs))