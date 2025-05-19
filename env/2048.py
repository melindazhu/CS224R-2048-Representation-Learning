import argparse
import logging
import random
import torch.nn as nn
import gymnasium as gym
import pygame

# https://github.com/Quentin18/gymnasium-2048/blob/main/scripts/train.py

logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

class Env_2048:
    def __init__(self, size, seed):
      
        self.env_name = "gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0"
        self.env = gym.make(self.env_name, size=size, render_mode="human")
        self.size = size
        self.seed = seed

        self.env.reset(seed=self.seed)
        self.logger = logger.info("Play game %s with size %d", self.env, self.size)

        self.total_score = 0
        self.current_board = None
        self.info = {}


    def step(self, action):

        '''
        actions = {
        0 : up
        1 : right
        2 : down
        3 : left
        }
        '''

        '''
        info dict contains: 
            board (2D array)
            step_score (int)
            total_score (int)
            max (int) <- number of tiles on board
            is_legal (bool)
            illegal_count (int)
        '''

        if action is not None:
            self.logger.info("action: %d", action)
            obs, reward, terminated, truncated, info = self.env.step(action)

        self.total_score = info["total_score"]

        self.logger.info("game over")
        self.logger.info("score: %d", info["total_score"])
        self.logger.info("max: %d", info["max"])

        self.current_board = info["board"]
        self.total_score = info["total_score"]

        return terminated, info, obs, reward


    def close(self):
        # Call after done with environment
        self.env.close()

    
    def get_score(self):
        return self.total_score

    def reset(self, seed=self.seed):
        self.env.reset(seed)

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
