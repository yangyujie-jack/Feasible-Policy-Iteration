from typing import Callable, Optional, Sequence

import haiku as hk
import jax
import jax.numpy as jnp

# Typings
Activation = Callable[[jnp.ndarray], jnp.ndarray]

# Constants
MIN_LOG_STD = -20
MAX_LOG_STD = 2

# Layers
Identity: Activation = lambda x: x


class QNet(hk.Module):
    def __init__(
        self,
        hidden_sizes: Sequence[int],
        activation: Activation = jax.nn.relu,
        output_activation: Activation = Identity,
        name: Optional[str] = None,
    ):
        super(QNet, self).__init__(name=name)
        self.fc = mlp(hidden_sizes, 1, activation, output_activation)

    def __call__(self, obs: jnp.ndarray, act: jnp.ndarray) -> jnp.ndarray:
        return self.fc(jnp.concatenate((obs, act), axis=-1)).squeeze(-1)


class StochasticPolicyNet(hk.Module):
    def __init__(
        self,
        act_dim: int,
        hidden_sizes: Sequence[int],
        activation: Activation = jax.nn.relu,
        output_activation: Activation = Identity,
        name: Optional[str] = None,
    ):
        super(StochasticPolicyNet, self).__init__(name=name)
        self.fc = mlp(hidden_sizes, act_dim * 2, activation, output_activation)

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        mean, log_std = jnp.split(self.fc(obs), 2, axis=-1)
        std = jnp.exp(jnp.clip(log_std, MIN_LOG_STD, MAX_LOG_STD))
        return mean, std


def mlp(
    hidden_sizes: Sequence[int],
    output_size: int,
    activation: Activation,
    output_activation: Activation,
):
    layers = []
    for hidden_size in hidden_sizes:
        layers += [hk.Linear(hidden_size), activation]
    layers += [hk.Linear(output_size), output_activation]
    return hk.Sequential(layers)
