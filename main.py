import argparse
from dqn2048.agent.dqn_agent import DQNAgent
from dqn2048.env.env_2048 import Env_2048
from dqn2048.train import train
from dqn2048.evals import plot_gini_vs_reward_hexbin, plot_gini_vs_reward_heatmap

def parse_args():
    parser = argparse.ArgumentParser()

    # Add any config values you want to tune
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.05)
    parser.add_argument('--epsilon_decay', type=int, default=20000)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--target_update_freq', type=int, default=100)
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='runs/2048')
    # auxiliary tasks and reward shaping parameters
    parser.add_argument('--legal-aux-weight', type=float, default=0.05)
    parser.add_argument('--max-tile-aux-weight', type=float, default=0.05)
    parser.add_argument('--illegal-move-penalty', type=float, default=1.0)
    parser.add_argument('--reward-shaping-max-tile-weight', type=float, default=1.0)
    parser.add_argument('--compression-shaping-weight', type=float, default=1.0)

    return parser.parse_args()

def main():
    args = parse_args()
    env = Env_2048(size=args.size, seed=args.seed)

    obs = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = 4

    config = vars(args)

    agent = DQNAgent(state_dim, action_dim, config)
    rewards, total_scores, max_tiles, illegal_move_counts, gini_values = train(env, agent, config)
    env.close()

    plot_gini_vs_reward_hexbin(gini_values, rewards)
    plot_gini_vs_reward_heatmap(gini_values, rewards)

if __name__ == "__main__":
    main()
