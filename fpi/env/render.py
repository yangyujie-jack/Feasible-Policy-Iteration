from typing import Optional

import gymnasium
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from fpi.agent.base import Agent


DELTA = 0.01


class SafetyGymnasiumRender:
    def __init__(
        self,
        ax: Axes,
        env: gymnasium.Env,
        agent: Agent,
        render_lidar: bool = False,
        frame_skip: int = 1,
        seed: Optional[int] = None,
    ):
        extents = env.task.placements_conf.extents
        ax.set_xlim(extents[0], extents[2])
        ax.set_ylim(extents[1], extents[3])
        ax.set_aspect('equal', 'box')

        self.obs, _ = env.reset(seed=seed)

        self.hazards_circles = []
        for pos in env.task.hazards.pos:
            circle = Circle(pos[:2], env.task.hazards.size, facecolor='b', alpha=0.5, zorder=0)
            self.hazards_circles.append(circle)
            ax.add_patch(circle)

        self.goal_circle = ax.add_patch(Circle(
            env.task.goal.pos[:2], env.task.goal.size, facecolor='g', alpha=0.5, zorder=0))

        self.robot_circle = ax.add_patch(Circle([0, 0], 0.05, facecolor='r', zorder=1))
        self.robot_head = ax.add_patch(Circle([0, 0], 0.02, facecolor='k', zorder=1))

        self.vio_circle = ax.add_patch(Circle([0, 0], 0.2, facecolor='r', alpha=0.0, zorder=1))

        if render_lidar:
            self.lidar_lines = []
            for _ in range(env.task.lidar_conf.num_bins):
                line = Line2D([0, 0], [0, 0], color='k', linewidth=0.5, alpha=0.5, zorder=0)
                self.lidar_lines.append(line)
                ax.add_line(line)

        self.g_text = ax.text(DELTA, 1 - DELTA, '', ha='left', va='top', transform=ax.transAxes)

        self.ax = ax
        self.env = env
        self.agent = agent
        self.render_lidar = render_lidar
        self.frame_skip = frame_skip
        self.done = False

    def update(self):
        violate = False
        if not self.done:
            for _ in range(self.frame_skip):
                action = self.agent.get_deterministic_action(self.obs)
                self.obs, _, cost, terminated, truncated, _ = self.env.step(action)
                violate = violate or (cost > 0)
                self.done = terminated or truncated
                if self.done:
                    break

        agent_pos = self.env.task.agent.pos[:2]
        self.goal_circle.set_center(self.env.task.goal.pos[:2])
        self.robot_circle.set_center(agent_pos)
        agent_angle = np.arctan2(self.env.task.agent.mat[1, 0], self.env.task.agent.mat[0, 0])
        self.robot_head.set_center([agent_pos[0] + 0.05 * np.cos(agent_angle),
                                    agent_pos[1] + 0.05 * np.sin(agent_angle)])
        self.vio_circle.set_center(agent_pos)
        if violate:
            self.vio_circle.set_alpha(0.5)
        else:
            self.vio_circle.set_alpha(0.0)

        if self.render_lidar:
            hazards_lidar = self.obs[-self.env.task.lidar_conf.num_bins:]
            hazards_dist = np.abs(np.log(np.maximum(hazards_lidar, 1e-3)))
            for i in range(self.env.task.lidar_conf.num_bins):
                line = self.lidar_lines[i]
                dist = hazards_dist[i]
                angle = agent_angle + 2 * np.pi * i / self.env.task.lidar_conf.num_bins + np.pi
                line.set_xdata([agent_pos[0], agent_pos[0] + dist * np.cos(angle)])
                line.set_ydata([agent_pos[1], agent_pos[1] + dist * np.sin(angle)])

        action = self.agent.get_deterministic_action(self.obs)
        g = self.agent.get_feasibility(self.obs, action)
        self.g_text.set_text('G: %.4f' % g)

        return self.robot_circle,
