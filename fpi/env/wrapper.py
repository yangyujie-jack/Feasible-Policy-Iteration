from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


class ConstraintInfo(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        constraint_low: float = -np.inf,
        constraint_high: float = np.inf,
    ):
        super().__init__(env)
        self.constraint_low = constraint_low
        self.constraint_high = constraint_high

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, cost, terminated, truncated, info = self.env.step(action)

        constraint = -np.inf
        task = self.env.unwrapped.task
        for h_pos in task.hazards.pos:
            h_dist = task.agent.dist_xy(h_pos)
            constraint = max(task.hazards.size - h_dist, constraint)
        info['constraint'] = float(np.clip(constraint, self.constraint_low, self.constraint_high))

        return obs, reward, cost, terminated, truncated, info
