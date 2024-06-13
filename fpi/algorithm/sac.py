import math
from typing import NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from fpi.agent.sac import SACAgent, SACParams
from fpi.algorithm.base import Algorithm
from fpi.utils.experience import Experience


class SACAlgState(NamedTuple):
    q1_opt_state: optax.OptState
    q2_opt_state: optax.OptState
    policy_opt_state: optax.OptState
    log_alpha: jnp.ndarray
    log_alpha_opt_state: optax.OptState


class SAC(Algorithm):
    def __init__(
        self,
        agent: SACAgent,
        *,
        gamma: float = 0.99,
        lr: float = 3e-4,
        tau: float = 0.005,
        alpha: float = 1.0,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        log_alpha = jnp.array(math.log(alpha), dtype=jnp.float32)
        if target_entropy is None:
            self.target_entropy = -self.agent.act_dim
        else:
            self.target_entropy = target_entropy
        self.optim = optax.adam(lr)
        self.alg_state = SACAlgState(
            q1_opt_state=self.optim.init(agent.params.q1),
            q2_opt_state=self.optim.init(agent.params.q2),
            policy_opt_state=self.optim.init(agent.params.policy),
            log_alpha=log_alpha,
            log_alpha_opt_state=self.optim.init(log_alpha),
        )

        @jax.jit
        def stateless_update(
            key: jax.random.KeyArray,
            params: SACParams,
            alg_state: SACAlgState,
            data: Experience
        ) -> Tuple[SACParams, SACAlgState, dict]:
            obs, action, reward, next_obs, done = (
                data.obs,
                data.action,
                data.reward,
                data.next_obs,
                data.done,
            )
            (
                q1_params,
                q2_params,
                target_q1_params,
                target_q2_params,
                policy_params,
            ) = params
            (
                q1_opt_state,
                q2_opt_state,
                policy_opt_state,
                log_alpha,
                log_alpha_opt_state,
            ) = alg_state
            next_eval_key, new_eval_key = jax.random.split(key)

            # update q
            next_action, next_logp = self.agent.evaluate(
                next_eval_key, policy_params, next_obs)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target) - jnp.exp(log_alpha) * next_logp
            q_backup = reward + (1 - done) * self.gamma * q_target

            def q_loss_fn(q_params: hk.Params):
                q = self.agent.q(q_params, obs, action)
                q_loss = ((q - q_backup) ** 2).mean()
                return q_loss, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(
                q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(
                q_loss_fn, has_aux=True)(q2_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            # update policy
            def policy_loss_fn(policy_params: hk.Params):
                new_action, new_logp = self.agent.evaluate(
                    new_eval_key, policy_params, obs)
                q1 = self.agent.q(q1_params, obs, new_action)
                q2 = self.agent.q(q2_params, obs, new_action)
                q = jnp.minimum(q1, q2)
                policy_loss = (jnp.exp(log_alpha) * new_logp - q).mean()
                return policy_loss, new_logp

            (policy_loss, new_logp), policy_grads = jax.value_and_grad(
                policy_loss_fn, has_aux=True)(policy_params)
            policy_update, policy_opt_state = self.optim.update(
                policy_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, policy_update)

            # update alpha
            log_alpha, log_alpha_opt_state = self.update_alpha(
                log_alpha, log_alpha_opt_state, new_logp)

            # update target networks
            target_q1_params = optax.incremental_update(
                q1_params, target_q1_params, self.tau)
            target_q2_params = optax.incremental_update(
                q2_params, target_q2_params, self.tau)

            params = SACParams(
                q1=q1_params,
                q2=q2_params,
                target_q1=target_q1_params,
                target_q2=target_q2_params,
                policy=policy_params,
            )
            alg_state = SACAlgState(
                q1_opt_state=q1_opt_state,
                q2_opt_state=q2_opt_state,
                policy_opt_state=policy_opt_state,
                log_alpha=log_alpha,
                log_alpha_opt_state=log_alpha_opt_state,
            )
            info = {
                'q1_loss': q1_loss,
                'q2_loss': q2_loss,
                'q1': q1.mean(),
                'q2': q2.mean(),
                'policy_loss': policy_loss,
                'entropy': -new_logp.mean(),
                'alpha': jnp.exp(log_alpha),
            }
            return params, alg_state, info

        self.stateless_update = stateless_update

    def update_alpha(
        self,
        log_alpha: jnp.ndarray,
        log_alpha_opt_state: optax.OptState,
        log_prob: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, optax.OptState]:
        def log_alpha_loss_fn(log_alpha: jnp.ndarray) -> jnp.ndarray:
            return -(log_alpha * (log_prob + self.target_entropy)).mean()

        def log_alpha_update_fn(
            log_alpha: jnp.ndarray,
            log_alpha_opt_state: optax.OptState,
        ) -> Tuple[jnp.ndarray, optax.OptState]:
            log_alpha_grads = jax.grad(log_alpha_loss_fn)(log_alpha)
            log_alpha_update, log_alpha_opt_state = self.optim.update(
                log_alpha_grads, log_alpha_opt_state)
            log_alpha = optax.apply_updates(log_alpha, log_alpha_update)
            return log_alpha, log_alpha_opt_state

        return jax.lax.cond(
            self.auto_alpha,
            log_alpha_update_fn,
            lambda *x: x,
            log_alpha,
            log_alpha_opt_state,
        )
