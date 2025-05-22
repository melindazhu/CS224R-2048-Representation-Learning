import numpy as np
from dqn2048.agent.dqn_agent import DQNAgent


def train(env, agent, config):
    rewards = []
    total_scores = []

    for episode in range(config['num_episodes']):
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
            total_reward += reward
            done = terminated

        if episode % config['target_update_freq'] == 0:
            agent.update_target()

        rewards.append(total_reward)
        total_scores.append(info['total_score'])
        print(f"Episode {episode}, Total Score: {info['total_score']}, Max Tile: {info['max']}")

    np.save("episode_scores.npy", total_scores)
    return rewards