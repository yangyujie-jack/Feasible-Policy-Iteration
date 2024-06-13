from typing import Callable, Optional

import gymnasium as gym
import numpy as np


def evaluate_one_episode(
    env: gym.Env,
    get_action: Callable[[np.ndarray], np.ndarray],
    seed: Optional[int] = None,
    options: Optional[dict] = None,
) -> dict:
    ep_ret, ep_cost, ep_len = 0.0, 0.0, 0

    obs, _ = env.reset(seed=seed, options=options)

    while True:
        action = get_action(obs)

        obs, reward, cost, terminated, truncated, _ = env.step(action)

        ep_ret += reward
        ep_cost += cost
        ep_len += 1

        if terminated or truncated:
            break

    return {
        'episode_return': ep_ret,
        'episode_cost': ep_cost,
        'episode_length': ep_len,
    }
