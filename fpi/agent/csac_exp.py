import math
from typing import NamedTuple, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.distributions import Normal

from fpi.agent.block import QNet, StochasticPolicyNet
from fpi.agent.sac import SACAgent


EPSILON = 1e-6


class CSACExpParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    g1: hk.Params
    g2: hk.Params
    target_g1: hk.Params
    target_g2: hk.Params
    policy: hk.Params
    exp_policy: hk.Params


class CSACExpAgent(SACAgent):
    def __init__(
        self,
        key: jax.random.KeyArray,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
    ):
        def q_fn(obs, act):
            return QNet(hidden_sizes)(obs, act)

        def policy_fn(obs):
            return StochasticPolicyNet(act_dim, hidden_sizes)(obs)

        q = hk.without_apply_rng(hk.transform(q_fn))
        policy = hk.without_apply_rng(hk.transform(policy_fn))

        q1_key, q2_key, g1_key, g2_key, policy_key, exp_policy_key = jax.random.split(key, 6)
        obs = jnp.zeros((1, obs_dim))
        act = jnp.zeros((1, act_dim))
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        g1_params = q.init(g1_key, obs, act)
        g2_params = q.init(g2_key, obs, act)
        target_g1_params = g1_params
        target_g2_params = g2_params
        policy_params = policy.init(policy_key, obs)
        exp_policy_params = policy.init(exp_policy_key, obs)
        self.params = CSACExpParams(
            q1=q1_params,
            q2=q2_params,
            target_q1=target_q1_params,
            target_q2=target_q2_params,
            g1=g1_params,
            g2=g2_params,
            target_g1=target_g1_params,
            target_g2=target_g2_params,
            policy=policy_params,
            exp_policy=exp_policy_params,
        )

        self.q = q.apply
        self.policy = policy.apply
        self.act_dim = act_dim

    def get_action(self, key: jax.random.KeyArray, obs: np.ndarray) -> np.ndarray:
        mean, std = self.policy(self.params.exp_policy, obs)
        z = Normal(mean, std).sample(key)
        return np.asarray(jnp.tanh(z))

    def evaluate_action(
        self, policy_params: hk.Params, obs: np.ndarray, act: np.ndarray
    ) -> np.ndarray:
        mean, std = self.policy(policy_params, obs)
        dist = Normal(mean, std)
        z = jnp.arctanh(jnp.clip(act, -1 + EPSILON, 1 - EPSILON))
        logp = (dist.log_prob(z) - 2 * (math.log(2) -
                z - jax.nn.softplus(-2 * z))).sum(axis=-1)
        return logp
