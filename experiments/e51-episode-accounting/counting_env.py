"""Counting-v0: step t of an episode pays reward t; the episode terminates at t = 5.

So every episode's return is 15 and its length 5. Importing this module
registers it, which lets another process make it as
gym.make("counting_env:Counting-v0").
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

LENGTH = 5


class Counting(gym.Env):
    observation_space = gym.spaces.Box(0.0, float(LENGTH), (1,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([0.0], np.float32), {}

    def step(self, action):
        self.t += 1
        obs = np.array([float(self.t)], np.float32)
        return obs, float(self.t), self.t == LENGTH, False, {}


gym.register("Counting-v0", entry_point=Counting)
