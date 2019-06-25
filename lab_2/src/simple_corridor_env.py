import numpy as np
import gym
from gym.spaces import Discrete, Box

class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config={'corridor_length': 5}):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(
            0.0, self.end_pos, shape=(1, ), dtype=np.float32)

    def reset(self):
        self.cur_pos = 0
        self.steps = 0
        return [self.cur_pos]

    def step(self, action):
        self.steps += 1
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos
        if done: #reached end of corridor
            reward = 10
        else:
            reward = 1
            
        if self.steps >= self.end_pos*10: #timeout
            done = True
            reward = -1
        observation = [self.cur_pos]
        return observation, reward, done, {}
    
    def render(self, mode):
        pass