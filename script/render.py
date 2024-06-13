import argparse
import json
import os

import jax
import safety_gymnasium
import matplotlib.pyplot as plt
import fpi.env
from fpi.agent.sac import SACAgent
from fpi.agent.csac import CSACAgent
from fpi.env.render import SafetyGymnasiumRender
from matplotlib.animation import FuncAnimation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl_path', type=str, default=None)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--render_lidar', action='store_true', default=False)
    parser.add_argument('--frame_skip', type=int, default=1)
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--seed', type=int, default=None)
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

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    render = SafetyGymnasiumRender(
        ax=ax,
        env=env,
        agent=agent,
        render_lidar=args.render_lidar,
        frame_skip=args.frame_skip,
        seed=args.seed,
    )
    ani = FuncAnimation(
        fig=fig,
        func=lambda _: render.update(),
        frames=int(args.episode_length / args.frame_skip),
        interval=args.interval,
        blit=True,
        cache_frame_data=False,
    )
    iter_num = args.pkl_path.split('/')[-1][:-4].split('_')[1]
    ani.save(os.path.join(os.path.dirname(args.pkl_path), f'{iter_num}_{args.seed}.mp4'))
