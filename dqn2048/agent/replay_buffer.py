import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Assert that state and next_state are 1D arrays of size 16 (i.e. flattened 4x4 board)
        assert isinstance(state, np.ndarray), f"Expected state to be np.ndarray, got {type(state)}"
        assert state.ndim == 1 and state.shape[0] == 16, f"Invalid state shape: {state.shape}"
        assert isinstance(next_state, np.ndarray), f"Expected next_state to be np.ndarray, got {type(next_state)}"
        assert next_state.ndim == 1 and next_state.shape[0] == 16, f"Invalid next_state shape: {next_state.shape}"

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        # Add assertion to make sure states are (batch_size, 16)
        assert states.ndim == 2 and states.shape[1] == 16, f"Bad shape in replay buffer sample: {states.shape}"
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
