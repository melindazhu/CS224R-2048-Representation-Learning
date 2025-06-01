import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    # push method for adding auxiliary tasks; uncomment the above if not using auxiliary tasks
    def push(self, state, action, reward, next_state, done, legal_vec, max_tile_log2=None):
        data = (state, action, reward, next_state, done, legal_vec, max_tile_log2)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity
    
    # sample method for auxiliary tasks
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, legal_vecs, max_tile_log2 = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones, legal_vecs, max_tile_log2

    def __len__(self):
        return len(self.buffer)
