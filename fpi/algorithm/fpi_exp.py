import math
from typing import NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from fpi.agent.csac_exp import CSACExpAgent, CSACExpParams
from fpi.algorithm.sac import SAC
from fpi.utils.experience import Experience
from fpi.utils.math import masked_mean


EPSILON = 1e-6


class FPIExpAlgState(NamedTuple):
    q1_opt_state: optax.OptState
    q2_opt_state: optax.OptState
    g1_opt_state: optax.OptState
    g2_opt_state: optax.OptState
    policy_opt_state: optax.OptState
    exp_policy_opt_state: optax.OptState
    log_alpha: jnp.ndarray
    log_alpha_opt_state: optax.OptState
    # log_alpha_exp: jnp.ndarray
    # log_alpha_exp_opt_state: optax.OptState


class FPIExp(SAC):
    def __init__(
        self,
        agent: CSACExpAgent,
        *,
        gamma: float = 0.99,
        lr: float = 3e-4,
        tau: float = 0.005,
        alpha: float = 1.0,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        pf: float = 0.1,
        t: float = 1.0,
        kl_penalty: float = 1.0,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        log_alpha = jnp.array(math.log(alpha), dtype=jnp.float32)
        # log_alpha_exp = jnp.array(math.log(alpha), dtype=jnp.float32)
        if target_entropy is None:
            self.target_entropy = -self.agent.act_dim
        else:
            self.target_entropy = target_entropy
        self.pf = pf
        self.t = t
        self.kl_penalty = kl_penalty
        self.optim = optax.adam(lr)
        self.alg_state = FPIExpAlgState(
            q1_opt_state=self.optim.init(agent.params.q1),
            q2_opt_state=self.optim.init(agent.params.q2),
            g1_opt_state=self.optim.init(agent.params.g1),
            g2_opt_state=self.optim.init(agent.params.g2),
            policy_opt_state=self.optim.init(agent.params.policy),
            exp_policy_opt_state=self.optim.init(agent.params.exp_policy),
            log_alpha=log_alpha,
            log_alpha_opt_state=self.optim.init(log_alpha),
            # log_alpha_exp=log_alpha_exp,
            # log_alpha_exp_opt_state=self.optim.init(log_alpha_exp),
        )

        @jax.jit
        def stateless_update(
            key: jax.random.KeyArray,
            params: CSACExpParams,
            alg_state: FPIExpAlgState,
            data: Experience
        ) -> Tuple[CSACExpParams, FPIExpAlgState, dict]:
            obs, action, next_obs, reward, cost, done = (
                data.obs,
                data.action,
                data.next_obs,
                data.reward,
                data.cost,
                data.done,
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
                exp_policy_params,
            ) = params
            (
                q1_opt_state,
                q2_opt_state,
                g1_opt_state,
                g2_opt_state,
                policy_opt_state,
                exp_policy_opt_state,
                log_alpha,
                log_alpha_opt_state,
                # log_alpha_exp,
                # log_alpha_exp_opt_state,
            ) = alg_state
            key_q, key_g, key_policy, key_exp_policy = jax.random.split(key, 4)

            # update q
            next_action, _ = self.agent.evaluate(key_q, policy_params, next_obs)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target)
            q_backup = reward + (1 - done) * self.gamma * q_target

            def q_loss_fn(q_params: hk.Params):
                q = self.agent.q(q_params, obs, action)
                q_loss = ((q - q_backup) ** 2).mean()
                return q_loss, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            # update g
            next_action, _ = self.agent.evaluate(key_g, policy_params, next_obs)
            g1_target = self.agent.q(target_g1_params, next_obs, next_action)
            g2_target = self.agent.q(target_g2_params, next_obs, next_action)
            g_target = jnp.clip(jnp.maximum(g1_target, g2_target), 0.0, 1.0)
            cost = (cost > 0).astype(jnp.float32)
            g_backup = cost + (1 - done) * (1 - cost) * self.gamma * g_target

            def g_loss_fn(g_params: hk.Params) -> jnp.ndarray:
                g = self.agent.q(g_params, obs, action)
                loss = ((g - g_backup) ** 2).mean()
                return loss, g

            (g1_loss, g1), g1_grads = jax.value_and_grad(g_loss_fn, has_aux=True)(g1_params)
            (g2_loss, g2), g2_grads = jax.value_and_grad(g_loss_fn, has_aux=True)(g2_params)
            g1_update, g1_opt_state = self.optim.update(g1_grads, g1_opt_state)
            g2_update, g2_opt_state = self.optim.update(g2_grads, g2_opt_state)
            g1_params = optax.apply_updates(g1_params, g1_update)
            g2_params = optax.apply_updates(g2_params, g2_update)

            # update policy
            def policy_loss_fn(policy_params: hk.Params):
                new_action, new_logp = self.agent.evaluate(key_policy, policy_params, obs)

                q1 = self.agent.q(q1_params, obs, new_action)
                q2 = self.agent.q(q2_params, obs, new_action)
                q = jnp.minimum(q1, q2)

                g1 = self.agent.q(g1_params, obs, new_action)
                g2 = self.agent.q(g2_params, obs, new_action)
                g = jnp.maximum(g1, g2)
                feas = g - self.pf < -EPSILON

                e = -jnp.exp(log_alpha) * new_logp

                feas_cons = jnp.clip(g - self.pf, -self.pf, -EPSILON)
                b_coef = self.t * (-1 / jax.lax.stop_gradient(feas_cons) - 1 / self.pf)
                feas_loss = (feas * (-q + b_coef * feas_cons) / (1 + b_coef) - e).mean()
                beta_feas = masked_mean(b_coef / (1 + b_coef), feas)

                infe_loss = (~feas * g).mean()

                return jnp.asarray([feas_loss, infe_loss]), (new_action, new_logp, feas, beta_feas)

            policy_loss_jac, aux = jax.jacrev(policy_loss_fn, has_aux=True)(policy_params)
            policy_feas_grads = jax.tree_map(lambda x: x[0], policy_loss_jac)
            policy_infe_grads = jax.tree_map(lambda x: x[1], policy_loss_jac)
            new_action, new_logp, feas, beta_feas = aux

            policy_grads = jax.tree_map(lambda x, y: x + y, policy_feas_grads, policy_infe_grads)
            policy_update, policy_opt_state = self.optim.update(policy_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, policy_update)

            # update exploratory policy
            def exp_policy_loss_fn(exp_policy_params: hk.Params):
                new_action_exp, new_logp_exp = self.agent.evaluate(key_exp_policy, exp_policy_params, obs)

                q1 = self.agent.q(q1_params, obs, new_action_exp)
                q2 = self.agent.q(q2_params, obs, new_action_exp)
                q = jnp.minimum(q1, q2)

                eval_logp = self.agent.evaluate_action(exp_policy_params, obs, new_action)
                kl = new_logp - eval_logp

                loss = (-q + self.kl_penalty * kl).mean()

                return loss, (new_logp_exp, kl)

            exp_policy_grads, aux = jax.grad(exp_policy_loss_fn, has_aux=True)(exp_policy_params)
            exp_policy_update, exp_policy_opt_state = self.optim.update(
                exp_policy_grads, exp_policy_opt_state)
            exp_policy_params = optax.apply_updates(exp_policy_params, exp_policy_update)
            new_logp_exp, kl = aux

            # update alpha
            log_alpha, log_alpha_opt_state = self.update_alpha(
                log_alpha, log_alpha_opt_state, new_logp)
            # log_alpha_exp, log_alpha_exp_opt_state = self.update_alpha(
            #     log_alpha_exp, log_alpha_exp_opt_state, new_logp_exp)

            # update target networks
            target_q1_params = optax.incremental_update(q1_params, target_q1_params, self.tau)
            target_q2_params = optax.incremental_update(q2_params, target_q2_params, self.tau)
            target_g1_params = optax.incremental_update(g1_params, target_g1_params, self.tau)
            target_g2_params = optax.incremental_update(g2_params, target_g2_params, self.tau)

            params = CSACExpParams(
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
            alg_state = FPIExpAlgState(
                q1_opt_state=q1_opt_state,
                q2_opt_state=q2_opt_state,
                g1_opt_state=g1_opt_state,
                g2_opt_state=g2_opt_state,
                policy_opt_state=policy_opt_state,
                exp_policy_opt_state=exp_policy_opt_state,
                log_alpha=log_alpha,
                log_alpha_opt_state=log_alpha_opt_state,
                # log_alpha_exp=log_alpha_exp,
                # log_alpha_exp_opt_state=log_alpha_exp_opt_state,
            )
            info = {
                'q1_loss': q1_loss,
                'q2_loss': q2_loss,
                'q1': q1.mean(),
                'q2': q2.mean(),
                'g1_loss': g1_loss,
                'g2_loss': g2_loss,
                'g1': g1.mean(),
                'g2': g2.mean(),
                'feasible_ratio': feas.mean(),
                'beta_feasible': beta_feas,
                'entropy': -new_logp.mean(),
                'entropy_explore': -new_logp_exp.mean(),
                'kl_explore': kl.mean(),
                'alpha': jnp.exp(log_alpha),
                # 'alpha_explore': jnp.exp(log_alpha),
                't': t,
            }
            return params, alg_state, info

        self.stateless_update = stateless_update
