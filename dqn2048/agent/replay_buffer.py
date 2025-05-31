import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    # push method for adding auxiliary tasks; uncomment the above if not using auxiliary tasks
    def push(self, state, action, reward, next_state, done, legal_vec):
        self.buffer.append((state, action, reward, next_state, done, legal_vec))
    
    # sample method for auxiliary tasks
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, legal_vecs = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones, legal_vecs

    def __len__(self):
        return len(self.buffer)
