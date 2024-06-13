from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np


class Experience(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    next_obs: jnp.ndarray
    reward: jnp.ndarray
    cost: jnp.ndarray
    done: jnp.ndarray
    constraint: jnp.ndarray

    def batch_size(self) -> Optional[int]:
        try:
            if self.reward.ndim > 0:
                return self.reward.shape[0]
            else:
                return None
        except AttributeError:
            return None

    def __repr__(self):
        return f'Experience(size={self.batch_size()})'

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return Experience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            cost=np.zeros(leading_dims, dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool8),
            constraint=np.zeros(leading_dims, dtype=np.float32),
        )
