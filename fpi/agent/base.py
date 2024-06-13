import pickle
from abc import abstractmethod, ABCMeta
from typing import NamedTuple

import jax
import numpy as np
from numpyro.distributions import Uniform


class Agent(metaclass=ABCMeta):
    params: NamedTuple

    @abstractmethod
    def get_action(self, key: jax.random.KeyArray, obs: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        ...

    def get_random_action(self, key: jax.random.KeyArray, act_dim: int) -> np.ndarray:
        return np.asarray(Uniform(-1, 1).sample(key, (act_dim,)))

    def save(self, path: str) -> None:
        params = jax.device_get(self.params)
        with open(path, 'wb') as f:
            pickle.dump(params, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            params = pickle.load(f)
        self.params = jax.device_put(params)
