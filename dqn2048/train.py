import numpy as np
from dqn2048.agent.dqn_agent import DQNAgent
from tqdm import tqdm


def train(env, agent, config):
    rewards = []
    total_scores = []
    max_tiles = []
    illegal_move_counts = []

    for episode in trange(config['num_episodes'], desc="Training Episodes"):
        print(f"Episode {episode+1}/{config['num_episodes']}")
        obs = env.reset()
        state = obs.flatten()  # ensure 1D
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, _, info = env.step(action) # `truncated` will be set to False
            next_state = next_obs.flatten()

            agent.replay_buffer.push(state, action, reward, next_state, terminated)
            agent.train_step()

            state = next_state
            total_reward += float(reward)
            done = terminated

        if episode % config['target_update_freq'] == 0:
            agent.update_target()

        rewards.append(total_reward)
        total_scores.append(info['total_score'])
        max_tiles.append(info['max'])
        illegal_move_counts.append(info['illegal_count'])
        
        print(f"\nEpisode {episode}, Total Score: {info['total_score']}, Max Tile: {info['max']}\n")

    np.save("episode_scores.npy", total_scores)
    return rewards, total_scores, max_tiles, illegal_move_counts
