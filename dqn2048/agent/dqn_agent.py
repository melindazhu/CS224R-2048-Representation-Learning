import torch
import torch.nn as nn
import numpy as np
# from dqn2048.agent.qnet import QNetwork
from dqn2048.agent.qnet_cnn import QNetworkWithCNN
from dqn2048.agent.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.q_network = QNetworkWithCNN(state_dim, action_dim).to(self.device)
        self.target_network = QNetworkWithCNN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config['lr'])
        self.replay_buffer = ReplayBuffer(config['buffer_size'])

        self.config = config # added for auxiliary tasks, more flexibility in getting access to more params
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        self.batch_size = config['batch_size']
        self.total_steps = 0

        self.action_dim = action_dim

    def select_action(self, state, legal_mask=None):
        self.total_steps += 1
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
                        np.exp(-1. * self.total_steps / self.epsilon_decay)
        if np.random.rand() < eps_threshold or legal_mask is None:
            legal_actions = np.where(legal_mask == 1)[0] if legal_mask is not None else np.arange(self.action_dim)
            return np.random.choice(legal_actions)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values, _, _ = self.q_network(state)
        q_values = q_values.squeeze()
        q_values[legal_mask == 0] = -float('inf')
        return q_values.argmax().item()

    def train_step(self, writer=None, step=None):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, legal_targets, max_tile_targets = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        legal_targets = torch.FloatTensor(legal_targets).to(self.device)
        max_tile_targets = torch.FloatTensor(max_tile_targets).unsqueeze(1).to(self.device)

        q_values, legal_logits, max_tile_pred = self.q_network(states)
        q_values = q_values.gather(1, actions)

        with torch.no_grad():
            next_q_values, _, _ = self.target_network(next_states)
            next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            target = rewards + (1 - dones) * self.gamma * next_q_values

        q_loss = nn.functional.mse_loss(q_values, target.to(self.device))
        legal_loss = nn.functional.binary_cross_entropy_with_logits(legal_logits, legal_targets)
        max_tile_loss = nn.functional.mse_loss(max_tile_pred, max_tile_targets)
        total_loss = (
            q_loss
            + self.config.get('legal_aux_weight', 0.1) * legal_loss
            + self.config.get('max_tile_aux_weight', 0.1) * max_tile_loss
        )
        self.optimizer.zero_grad()
        total_loss.backward()
        # Add gradient clipping to stabilize loss
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        if writer and step is not None:
            writer.add_scalar('Loss/Q', q_loss.item(), step)
            writer.add_scalar('Loss/LegalAux', legal_loss.item(), step)
            writer.add_scalar('Loss/MaxTileAux', max_tile_loss.item(), step)
            writer.add_scalar('Loss/Total', total_loss.item(), step)

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
