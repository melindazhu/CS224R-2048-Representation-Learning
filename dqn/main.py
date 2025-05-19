from dqn_agent import DQNAgent
from train import train
from env import Env_2048  # idk if this works, might need to restructure dirs

def main():
    size = 4
    seed = 42
    env = Env_2048(size=size, seed=seed)

    # get initial observation to determine input dimension
    obs = env.reset()
    state_dim = obs.flatten().shape[0]  # should be 16
    action_dim = 4  # up, right, down, left

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
