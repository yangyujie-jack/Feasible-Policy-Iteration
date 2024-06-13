import argparse
import json
import os

import jax
import pandas as pd
import safety_gymnasium
import fpi.env
from fpi.agent.sac import SACAgent
from fpi.agent.csac import CSACAgent
from fpi.utils.evaluate import evaluate_one_episode
from fpi.utils.random import seeding


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl_path', type=str, default=None)
    parser.add_argument('--episode_num', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(args.pkl_path), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    env = safety_gymnasium.make(config['env'])

    init_network_key = jax.random.PRNGKey(0)
    if config['alg'] == 'SAC':
        agent = SACAgent(
            key=init_network_key,
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            hidden_sizes=[config['hidden_dim']] * config['hidden_num'],
        )
    else:
        agent = CSACAgent(
            key=init_network_key,
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            hidden_sizes=[config['hidden_dim']] * config['hidden_num'],
        )
    agent.load(args.pkl_path)

    rng, _ = seeding(args.seed)

    res = {}
    for i in range(args.episode_num):
        seed = int(rng.integers(0, 2 ** 32 - 1))
        ep_info = {'seed': seed}
        ep_info.update(evaluate_one_episode(env, agent.get_deterministic_action, seed))
        for k, v in ep_info.items():
            if k in res:
                res[k].append(v)
            else:
                res[k] = [v]

    df = pd.DataFrame(res)
    iter_num = args.pkl_path.split('/')[-1][:-4].split('_')[1]
    df.to_csv(os.path.join(os.path.dirname(args.pkl_path), iter_num + '.csv'))
