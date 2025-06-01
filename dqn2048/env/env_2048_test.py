from env_2048 import SafeTwentyFortyEightEnv
import numpy as np

env = SafeTwentyFortyEightEnv(size=4)
obs, _ = env.reset()

# force a specific board state with two 128s ready to merge
env.board = np.array([
    [7, 7, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
])

obs, reward, terminated, truncated, info = env.step(1)  # right

print(env.board)
print("Reward:", reward)
print("Step score:", info["step_score"])
print("Max tile:", info["max"])
