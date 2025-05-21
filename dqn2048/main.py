from dqn2048.agent.dqn_agent import DQNAgent
from dqn2048.env.env_2048 import Env_2048
from dqn2048.train import train

def main():
    size = 4
    seed = 42
    env = Env_2048(size=size, seed=seed)

    obs = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = 4

    config = {
        'lr': 1e-3,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 10000,
        'buffer_size': 50000,
        'batch_size': 64,
        'target_update_freq': 10,
        'num_episodes': 500,
    }

    agent = DQNAgent(state_dim, action_dim, config)
    train(env, agent, config)
    env.close()

if __name__ == "__main__":
    main()
