import argparse
import logging
import random
import numpy as np
import torch.nn as nn
import gymnasium as gym
import pygame
from gymnasium_2048.envs.twenty_forty_eight import TwentyFortyEightEnv

# logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)
# logger = logging.getLogger(__name__)


# Wrapper around TwentyFortyEightEnv; inherits everything but changes apply_action
# to resolve the overflow warning
class SafeTwentyFortyEightEnv(TwentyFortyEightEnv):
    def apply_action(self, board, action):
        next_board, step_score, is_legal = super().apply_action(board, action)
        step_score = float(step_score)
        return next_board, step_score, is_legal


class Env_2048:
    def __init__(self, size, seed):
      
        self.env_name = "gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0"
        # self.env = gym.make(self.env_name, size=size, render_mode=None)
        # use the new wrapper instead
        self.env = SafeTwentyFortyEightEnv(size=size, render_mode=None)
        self.size = size
        self.seed = seed

        self.env.reset(seed=self.seed)
        print(f"Play game {self.env} with size {self.size}")

        self.total_score = 0
        self.current_board = None
        self.info = {}


    def step(self, action):
        '''
        actions = {
            0 : up,
            1 : right,
            2 : down,
            3 : left
        }

        info dict contains: 
            board (2D array)
            step_score (int)
            total_score (int)
            max (int) <- largest tile
            is_legal (bool)
            illegal_count (int)
        '''

        if action is not None:
            # we can probably remove this if it gets too verbose
            # print(f"Action taken: {action}")
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            obs, reward, terminated, truncated, info = self.env.step(0)  # default to up

        # Decode one-hot encoded obs (shape [4, 4, 16]) -> board of shape [4, 4] with actual tile values
        obs = np.argmax(obs, axis=-1)
        obs = np.where(obs > 0, 2 ** obs, 0)

        self.total_score = float(info["total_score"])
        info['total_score'] = self.total_score
        self.current_board = info["board"]
        info['step_score'] = float(info.get('step_score', 0))
        self.info = info

        # we can probably remove this if it gets too verbose
        # print(f"Step score: {info['step_score']}, Total score: {info['total_score']}, Max tile: {info['max']}")

        # only print game-over message when episode ends
        if terminated:
            print("Game over")
            print(f"Final score: {info['total_score']}")
            print(f"Largest tile reached: {info['max']}")
            print(f"Illegal move counts: {info['illegal_count']}")

        return obs, reward, terminated, False, info

    def close(self):
        # Call after done with environment
        self.env.close()

    def get_score(self):
        return self.total_score

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed or self.seed)

        # Decode one-hot encoded tiles back to board values
        obs = np.argmax(obs, axis=-1)  # shape: (4, 4), values: exponents
        obs = np.where(obs > 0, 2 ** obs, 0)  # convert to actual tile values
        return obs

    def get_current_info(self):

        '''
        info dict contains: 
            board (2D array)
            step_score (int)
            total_score (int)
            max (int) <- number of tiles on board
            is_legal (bool)
            illegal_count (int)
        '''

        return self.info

    def get_legal_vector(self):
        board = self.env.unwrapped.board.copy()
        legal = []

        # https://github.com/Quentin18/gymnasium-2048/blob/dea2448066e88198f87e4767cbe34c3f5ffcd8db/src/gymnasium_2048/envs/twenty_forty_eight.py#L88
        for action in range(4):
            _, _, is_legal = self.env.unwrapped.apply_action(board, action)
            legal.append(int(is_legal))

        return legal
