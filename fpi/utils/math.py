from typing import Optional, Tuple, Union

import jax.numpy as jnp


def masked_mean(
    x: jnp.ndarray,
    mask: jnp.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> jnp.ndarray:
    return (mask * x).sum(axis) / jnp.maximum(mask.sum(axis), 1)
