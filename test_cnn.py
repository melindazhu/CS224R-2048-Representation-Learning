from dqn2048.env.env_2048 import Env_2048

env = Env_2048(size=4, seed=42)
obs = env.reset()
print("obs shape:", obs.shape)
print("obs:\n", obs)
