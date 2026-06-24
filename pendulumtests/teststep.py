from dm_control import suite
import numpy as np

env = suite.load("cheetah", "run")   # your domain/task here

print("ACTION:", env.action_spec())
print("OBS:")
for k, v in env.observation_spec().items():
    print(f"  {k:<14} shape={v.shape} dtype={v.dtype}")

ts = env.reset()
print("reset obs:", {k: np.asarray(v).ravel() for k, v in ts.observation.items()})

ts = env.step(np.zeros(env.action_spec().shape))
print("reward:", ts.reward, " discount:", ts.discount, " step_type:", ts.step_type)