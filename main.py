import argparse
from dqn2048.agent.dqn_agent import DQNAgent
from dqn2048.env.env_2048 import Env_2048
from dqn2048.train import train

def parse_args():
    parser = argparse.ArgumentParser()

    # Add any config values you want to tune
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.05)
    parser.add_argument('--epsilon_decay', type=int, default=20000)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--target_update_freq', type=int, default=100)
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int, default=4)

    return parser.parse_args()

def main():
    args = parse_args()
    env = Env_2048(size=args.size, seed=args.seed)

    obs = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = 4

    config = vars(args)

    agent = DQNAgent(state_dim, action_dim, config)
    train(env, agent, config)
    env.close()

if __name__ == "__main__":
    main()
