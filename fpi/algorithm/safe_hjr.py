import math
from typing import NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from fpi.agent.csac import CSACAgent, CSACParams
from fpi.algorithm.sac import SAC
from fpi.utils.experience import Experience


class SafeHJRAlgState(NamedTuple):
    g1_opt_state: optax.OptState
    g2_opt_state: optax.OptState
    policy_opt_state: optax.OptState
    log_alpha: jnp.ndarray
    log_alpha_opt_state: optax.OptState


class SafeHJR(SAC):
    def __init__(
        self,
        agent: CSACAgent,
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
        self.alg_state = SafeHJRAlgState(
            g1_opt_state=self.optim.init(agent.params.g1),
            g2_opt_state=self.optim.init(agent.params.g2),
            policy_opt_state=self.optim.init(agent.params.policy),
            log_alpha=log_alpha,
            log_alpha_opt_state=self.optim.init(log_alpha),
        )

        @jax.jit
        def stateless_update(
            key: jax.random.KeyArray,
            params: CSACParams,
            alg_state: SafeHJRAlgState,
            data: Experience
        ) -> Tuple[CSACParams, SafeHJRAlgState, dict]:
            obs, action, next_obs, done, constraint = (
                data.obs,
                data.action,
                data.next_obs,
                data.done,
                data.constraint,
            )
            (
                q1_params,
                q2_params,
                target_q1_params,
                target_q2_params,
                g1_params,
                g2_params,
                target_g1_params,
                target_g2_params,
                policy_params,
            ) = params
            (
                g1_opt_state,
                g2_opt_state,
                policy_opt_state,
                log_alpha,
                log_alpha_opt_state,
            ) = alg_state
            key_g, key_policy = jax.random.split(key, 2)

            # update g
            next_action, _ = self.agent.evaluate(key_g, policy_params, next_obs)
            g1_target = self.agent.q(target_g1_params, next_obs, next_action)
            g2_target = self.agent.q(target_g2_params, next_obs, next_action)
            g_target = jnp.maximum(g1_target, g2_target)
            g_backup = (1 - self.gamma) * constraint + \
                (1 - done) * self.gamma * jnp.maximum(constraint, g_target)

            def g_loss_fn(g_params: hk.Params) -> jnp.ndarray:
                g = self.agent.q(g_params, obs, action)
                loss = ((g - g_backup) ** 2).mean()
                return loss, g

            (g1_loss, g1), g1_grads = jax.value_and_grad(
                g_loss_fn, has_aux=True)(g1_params)
            (g2_loss, g2), g2_grads = jax.value_and_grad(
                g_loss_fn, has_aux=True)(g2_params)
            g1_update, g1_opt_state = self.optim.update(g1_grads, g1_opt_state)
            g2_update, g2_opt_state = self.optim.update(g2_grads, g2_opt_state)
            g1_params = optax.apply_updates(g1_params, g1_update)
            g2_params = optax.apply_updates(g2_params, g2_update)

            # update policy
            def policy_loss_fn(policy_params: hk.Params):
                new_action, new_logp = self.agent.evaluate(key_policy, policy_params, obs)
                g1 = self.agent.q(g1_params, obs, new_action)
                g2 = self.agent.q(g2_params, obs, new_action)
                g = jnp.maximum(g1, g2)
                loss = (g + jnp.exp(log_alpha) * new_logp).mean()
                return loss, new_logp

            (policy_loss, new_logp), policy_grads = jax.value_and_grad(
                policy_loss_fn, has_aux=True)(policy_params)
            policy_update, policy_opt_state = self.optim.update(
                policy_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, policy_update)

            # update alpha
            log_alpha, log_alpha_opt_state = self.update_alpha(
                log_alpha, log_alpha_opt_state, new_logp)

            # update target networks
            target_g1_params = optax.incremental_update(g1_params, target_g1_params, self.tau)
            target_g2_params = optax.incremental_update(g2_params, target_g2_params, self.tau)

            params = CSACParams(
                q1=q1_params,
                q2=q2_params,
                target_q1=target_q1_params,
                target_q2=target_q2_params,
                g1=g1_params,
                g2=g2_params,
                target_g1=target_g1_params,
                target_g2=target_g2_params,
                policy=policy_params,
            )
            alg_state = SafeHJRAlgState(
                g1_opt_state=g1_opt_state,
                g2_opt_state=g2_opt_state,
                policy_opt_state=policy_opt_state,
                log_alpha=log_alpha,
                log_alpha_opt_state=log_alpha_opt_state,
            )
            info = {
                'g1_loss': g1_loss,
                'g2_loss': g2_loss,
                'g1': g1.mean(),
                'g2': g2.mean(),
                'policy_loss': policy_loss,
                'entropy': -new_logp.mean(),
                'alpha': jnp.exp(log_alpha),
            }
            return params, alg_state, info

        self.stateless_update = stateless_update
