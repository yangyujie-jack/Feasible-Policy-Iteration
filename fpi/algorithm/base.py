import pickle
from typing import Callable, NamedTuple, Tuple

import jax

from fpi.agent.base import Agent
from fpi.utils.experience import Experience


class Algorithm:
    agent: Agent
    alg_state: NamedTuple
    stateless_update: Callable[[jax.random.KeyArray, NamedTuple, Experience],
                               Tuple[NamedTuple, dict]]

    def update(self, key: jax.random.KeyArray, data: Experience) -> dict:
        self.agent.params, self.alg_state, info = self.stateless_update(
            key, self.agent.params, self.alg_state, data)
        return info

    def save(self, path: str) -> None:
        alg_state = jax.device_get(self.alg_state)
        with open(path, 'wb') as f:
            pickle.dump(alg_state, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            alg_state = pickle.load(f)
        self.alg_state = jax.device_put(alg_state)
