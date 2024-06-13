import argparse
import json
import os
import time

import jax
import safety_gymnasium

import fpi.env
from fpi.env.wrapper import ConstraintInfo
from fpi.agent.sac import SACAgent
from fpi.agent.csac import CSACAgent
from fpi.agent.csac_exp import CSACExpAgent
from fpi.algorithm.sac import SAC
from fpi.algorithm.safe_cvf import SafeCVF
from fpi.algorithm.safe_cdf import SafeCDF
from fpi.algorithm.safe_hjr import SafeHJR
from fpi.algorithm.fpi_lin import FPILin
from fpi.algorithm.fpi_sig import FPISig
from fpi.algorithm.fpi_hjr import FPIHJR
from fpi.algorithm.fpi_mg import FPIMG
from fpi.algorithm.fpi_exp import FPIExp
from fpi.buffer.tree import TreeBuffer
from fpi.trainer.off_policy import OffPolicyTrainer
from fpi.utils.path import PROJECT_ROOT
from fpi.utils.random import seeding


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--env', type=str, default='CustomPointGoal1-v0')
    parser.add_argument('--alg', type=str, default='SAC')
    parser.add_argument('--hidden_num', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    parser.add_argument('--start_step', type=int, default=10000)
    parser.add_argument('--total_step', type=int, default=int(2e6))
    parser.add_argument('--sample_per_iteration', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=100000)
    parser.add_argument('--max_save_num', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--GPU_memory', type=str, default='.1')

    # SAC
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--fixed_alpha', action='store_true', default=False)

    # FPI
    parser.add_argument('--pf', type=float, default=0.1)
    parser.add_argument('--t', type=float, default=0.1)
    parser.add_argument('--min_grad_cos', type=float, default=0.8)
    parser.add_argument('--min_constraint', type=float, default=-1.0)
    parser.add_argument('--kl_penalty', type=float, default=0.01)

    args = parser.parse_args()

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = args.GPU_memory

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    init_network_seed, buffer_seed, train_seed = map(
        int, master_rng.integers(0, 2 ** 32 - 1, 3))

    env = ConstraintInfo(safety_gymnasium.make(args.env), constraint_low=-1.0)
    eval_env = safety_gymnasium.make(args.env)

    buffer = TreeBuffer.from_experience(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        size=args.buffer_size,
        seed=buffer_seed,
    )

    init_network_key = jax.random.PRNGKey(init_network_seed)

    if args.alg == 'SAC':
        agent = SACAgent(
            key=init_network_key,
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            hidden_sizes=[args.hidden_dim] * args.hidden_num,
        )
    elif args.alg == 'FPIExp':
        agent = CSACExpAgent(
            key=init_network_key,
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            hidden_sizes=[args.hidden_dim] * args.hidden_num,
        )
    else:
        agent = CSACAgent(
            key=init_network_key,
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            hidden_sizes=[args.hidden_dim] * args.hidden_num,
        )

    if args.alg == 'SAC':
        algorithm = SAC(
            agent,
            lr=args.lr,
            alpha=args.alpha,
            auto_alpha=not args.fixed_alpha,
        )
    elif args.alg == 'SafeCVF':
        algorithm = SafeCVF(
            agent,
            lr=args.lr,
            alpha=args.alpha,
            auto_alpha=not args.fixed_alpha,
        )
    elif args.alg == 'SafeCDF':
        algorithm = SafeCDF(
            agent,
            lr=args.lr,
            alpha=args.alpha,
            auto_alpha=not args.fixed_alpha,
        )
    elif args.alg == 'SafeHJR':
        algorithm = SafeHJR(
            agent,
            lr=args.lr,
            alpha=args.alpha,
            auto_alpha=not args.fixed_alpha,
        )
    elif args.alg == 'FPILin':
        algorithm = FPILin(
            agent,
            lr=args.lr,
            alpha=args.alpha,
            auto_alpha=not args.fixed_alpha,
            pf=args.pf,
            t=args.t,
        )
    elif args.alg == 'FPISig':
        algorithm = FPISig(
            agent,
            lr=args.lr,
            alpha=args.alpha,
            auto_alpha=not args.fixed_alpha,
            pf=args.pf,
            t=args.t,
        )
    elif args.alg == 'FPIHJR':
        algorithm = FPIHJR(
            agent,
            lr=args.lr,
            alpha=args.alpha,
            auto_alpha=not args.fixed_alpha,
            min_constraint=args.min_constraint,
            t=args.t,
        )
    elif args.alg == 'FPIMG':
        algorithm = FPIMG(
            agent,
            lr=args.lr,
            alpha=args.alpha,
            auto_alpha=not args.fixed_alpha,
            pf=args.pf,
            t=args.t,
            min_grad_cos=args.min_grad_cos,
        )
    elif args.alg == 'FPIExp':
        algorithm = FPIExp(
            agent,
            lr=args.lr,
            alpha=args.alpha,
            auto_alpha=not args.fixed_alpha,
            pf=args.pf,
            t=args.t,
            kl_penalty=args.kl_penalty,
        )
    else:
        raise ValueError(f'Invalid algorithm {args.alg}!')

    log_path = os.path.join(
        PROJECT_ROOT, 'log', args.env,
        args.alg + f'_seed{args.seed}_' + time.strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_path, exist_ok=True)

    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    trainer = OffPolicyTrainer(
        env=env,
        agent=agent,
        algorithm=algorithm,
        buffer=buffer,
        log_path=log_path,
        batch_size=args.batch_size,
        start_step=args.start_step,
        total_step=args.total_step,
        sample_per_iteration=args.sample_per_iteration,
        evaluate_env=eval_env,
        save_every=args.save_every,
        max_save_num=args.max_save_num,
    )

    trainer.train(train_seed)
