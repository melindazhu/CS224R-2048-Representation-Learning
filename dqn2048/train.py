import numpy as np
from dqn2048.agent.dqn_agent import DQNAgent
from tqdm import tqdm, trange
import gc # garbage collection
from torch.utils.tensorboard import SummaryWriter


def train(env, agent, config):
    writer = SummaryWriter(log_dir=config.get('log_dir', 'runs/2048_cnn_legal_aux'))
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
            reward = float(reward)
            if not info["is_legal"]:
                reward -= config.get("illegal_move_penalty", 2.0)
            legal_vec = env.get_legal_vector()
            agent.replay_buffer.push(state, action, reward, next_state, terminated, legal_vec)
            agent.train_step()

            state = next_state
            total_reward += float(reward)
            done = terminated

        if episode % config['target_update_freq'] == 0:
            agent.update_target()

        writer.add_scalar('Reward/Total', total_reward, episode) # might not care about this
        rewards.append(total_reward)
        writer.add_scalar('Score/Total', info['total_score'], episode)
        total_scores.append(info['total_score'])
        writer.add_scalar('Tile/Max', info['max'], episode)
        max_tiles.append(info['max'])
        writer.add_scalar('IllegalMoves/PerEpisode', info['illegal_count'], episode)
        illegal_move_counts.append(info['illegal_count'])
        
        print(f"\nEpisode {episode}, Total Score: {info['total_score']}, Max Tile: {info['max']}\n")

        if episode % 10 == 0:
            gc.collect()

    np.save("episode_scores.npy", total_scores)
    writer.close()
    return rewards, total_scores, max_tiles, illegal_move_counts
