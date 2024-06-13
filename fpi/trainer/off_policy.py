import os
from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
import jax
import numpy as np
from tensorboardX import SummaryWriter

from fpi.algorithm.base import Algorithm
from fpi.agent.base import Agent
from fpi.buffer.base import Buffer
from fpi.utils.evaluate import evaluate_one_episode
from fpi.utils.experience import Experience
from fpi.utils.random import seeding


class OffPolicyTrainer:
    def __init__(
        self,
        env: gym.Env,
        agent: Agent,
        algorithm: Algorithm,
        buffer: Buffer,
        log_path: str,
        batch_size: int = 256,
        start_step: int = 10000,
        total_step: int = int(1e6),
        sample_per_iteration: int = 1,
        update_per_iteration: int = 1,
        evaluate_env: Optional[gym.Env] = None,
        evaluate_every: int = 10000,
        evaluate_n_episode: int = 10,
        sample_log_n_episode: int = 10,
        update_log_n_step: int = 1000,
        save_every: int = 100000,
        max_save_num: int = 3,
    ):
        self.env = env
        self.agent = agent
        self.algorithm = algorithm
        self.buffer = buffer

        self.logger = SummaryWriter(log_path)
        self.log_path = log_path

        self.batch_size = batch_size
        self.start_step = start_step
        self.total_step = total_step
        self.sample_per_iteration = sample_per_iteration
        self.update_per_iteration = update_per_iteration
        self.evaluate_env = evaluate_env
        self.evaluate_every = evaluate_every
        self.evaluate_n_episode = evaluate_n_episode
        self.sample_log_n_episode = sample_log_n_episode
        self.update_log_n_step = update_log_n_step
        self.save_every = save_every
        self.max_save_num = max_save_num
        self.save_steps = []

    def train(self, seed: int):
        key = jax.random.PRNGKey(seed)
        iter_key_fn = create_iter_key_fn(key)
        rng, _ = seeding(seed)

        sample_step, sample_episode, update_step = 0, 0, 0
        ep_ret, ep_cost, ep_len = 0.0, 0.0, 0

        sample_info = {
            'episode_return': [],
            'episode_cost': [],
            'episode_length': [],
        }
        update_info: Dict[str, list] = {}

        action_space_seed = int(rng.integers(0, 2 ** 32 - 1))
        self.env.action_space.seed(action_space_seed)
        env_seed = int(rng.integers(0, 2 ** 32 - 1))
        obs, _ = self.env.reset(seed=env_seed)

        while sample_step < self.total_step:
            # setup random keys
            sample_key, update_key = iter_key_fn(sample_step)

            # sample data
            for _ in range(self.sample_per_iteration):
                if sample_step < self.start_step:
                    action = self.agent.get_random_action(
                        sample_key, self.env.action_space.shape[0])
                else:
                    action = self.agent.get_action(sample_key, obs)

                next_obs, reward, cost, terminated, truncated, info = self.env.step(action)

                experience = Experience(obs, action, next_obs, reward, cost, terminated,
                                        info['constraint'])
                self.buffer.add(experience)

                ep_ret += reward
                ep_cost += cost
                ep_len += 1
                sample_step += 1

                if terminated or truncated:
                    sample_info['episode_return'].append(ep_ret)
                    sample_info['episode_cost'].append(ep_cost)
                    sample_info['episode_length'].append(ep_len)
                    sample_episode += 1

                    if sample_episode % self.sample_log_n_episode == 0:
                        for k, v in sample_info.items():
                            self.logger.add_scalar(f'sample/{k}', np.mean(v), sample_step)
                            sample_info[k] = []
                        print('sample step', sample_step)

                    ep_ret, ep_cost, ep_len = 0.0, 0.0, 0

                    env_seed = int(rng.integers(0, 2 ** 32 - 1))
                    obs, _ = self.env.reset(seed=env_seed)
                else:
                    obs = next_obs

            if sample_step < self.start_step:
                continue

            # update parameters
            for _ in range(self.update_per_iteration):
                data = self.buffer.sample(self.batch_size)
                alg_info = self.algorithm.update(update_key, data)
                for k, v in alg_info.items():
                    if k in update_info:
                        update_info[k].append(v)
                    else:
                        update_info[k] = [v]

                update_step += 1

                if update_step % self.update_log_n_step == 0:
                    for k, v in update_info.items():
                        self.logger.add_scalar(f'update/{k}', np.mean(v), update_step)
                        update_info[k] = []
                    print('update step', update_step)

                if update_step % self.save_every == 0:
                    self.save(update_step)

            # evaluate
            if self.evaluate_env is not None and sample_step % self.evaluate_every == 0:
                self.evaluate(sample_step, rng)

        self.save(update_step)

    def evaluate(self, sample_step: int, rng: np.random.Generator):
        eval_info = {
            'episode_return': [],
            'episode_cost': [],
            'episode_length': [],
        }
        for _ in range(self.evaluate_n_episode):
            eval_env_seed = int(rng.integers(0, 2 ** 32 - 1))
            eval_ep_info = evaluate_one_episode(
                self.evaluate_env,
                self.agent.get_deterministic_action,
                eval_env_seed,
            )
            for k in eval_info.keys():
                eval_info[k].append(eval_ep_info[k])
        for k, v in eval_info.items():
            self.logger.add_scalar(f'evaluate/{k}', np.mean(v), sample_step)

    def save(self, step: int):
        self.agent.save(os.path.join(self.log_path, f'params_{step}.pkl'))
        self.save_steps.append(step)
        if len(self.save_steps) > self.max_save_num:
            remove_step = self.save_steps.pop(0)
            os.remove(os.path.join(self.log_path, f'params_{remove_step}.pkl'))


def create_iter_key_fn(key: jax.random.KeyArray) \
        -> Callable[[int], Tuple[jax.random.KeyArray, jax.random.KeyArray]]:
    def iter_key_fn(step: int):
        iter_key = jax.random.fold_in(key, step)
        sample_key, update_key = jax.random.split(iter_key)
        return sample_key, update_key

    iter_key_fn = jax.jit(iter_key_fn)
    iter_key_fn(0)  # Warm up
    return iter_key_fn
