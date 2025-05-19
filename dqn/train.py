import gym
import numpy as np
from dqn_agent import DQNAgent


def train(env, agent, config):
    rewards = []

    for episode in range(config['num_episodes']):
        obs = env.reset()
        state = obs.flatten()  # ensure 1D
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            terminated, info, next_obs, reward = env.step(action)
            next_state = next_obs.flatten()

            agent.replay_buffer.push(state, action, reward, next_state, terminated)
            agent.train_step()

            state = next_state
            total_reward += reward
            done = terminated

        if episode % config['target_update_freq'] == 0:
            agent.update_target()

        rewards.append(total_reward)
        print(f"Episode {episode}, Total Score: {info['total_score']}, Max Tile: {info['max']}")
    return rewards