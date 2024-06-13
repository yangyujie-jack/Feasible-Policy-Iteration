import math

import numpy as np
import gymnasium
from safety_gymnasium.assets.geoms import Goal, Hazards
from safety_gymnasium.bases.base_task import LidarConf
from safety_gymnasium.tasks import GoalLevel0


class CustomGoalLevel1(GoalLevel0):
    def __init__(self, config, lidar_num_bins: int = 32) -> None:
        super().__init__(config)

        self.lidar_conf = LidarConf(num_bins=lidar_num_bins)

        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]

        self._add_geoms(
            Goal(keepout=0.305, is_lidar_observed=False, is_comp_observed=True),
            Hazards(num=8, keepout=0.18),
        )

        self.last_dist_goal = None

    def build_observation_space(self) -> gymnasium.spaces.Dict:
        super().build_observation_space()

        self.obs_info.obs_space_dict['goal_comp'] = gymnasium.spaces.Box(
            low=np.asarray([-1.0] * self.compass_conf.shape + [0.0]),
            high=np.asarray([1.0] * self.compass_conf.shape + [1.0]),
            dtype=np.float64,
        )

        self.obs_info.obs_space_dict['hazards_lidar'] = gymnasium.spaces.Box(
            0.0,
            np.inf,
            (self.lidar_conf.num_bins,),
            dtype=np.float64,
        )

        if self.observation_flatten:
            self.observation_space = gymnasium.spaces.utils.flatten_space(
                self.obs_info.obs_space_dict,
            )
        else:
            self.observation_space = self.obs_info.obs_space_dict

    def _obs_compass(self, pos: np.ndarray) -> np.ndarray:
        angle = super()._obs_compass(pos)

        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.concatenate([pos, [0]])
        vec = pos - self.agent.pos
        vec = np.matmul(vec, self.agent.mat)
        vec = vec[: self.compass_conf.shape]
        dis = np.linalg.norm(vec, 2)
    
        com = np.append(angle, np.exp(-dis))
        return com

    def _obs_lidar_pseudo(self, positions: np.ndarray) -> np.ndarray:
        # This Lidar observation only suits hazards.
        r = self.hazards.size

        positions = np.asarray(positions)
        vec = positions[:, :2] - self.agent.pos[:2]
        vec = vec @ self.agent.mat[:2, :2]
        # distance from hazard center to robot
        dist_hr = np.linalg.norm(vec, 2, axis=1)
        dist_hr = dist_hr[:, np.newaxis]

        angle_hr = np.arctan2(vec[:, 1], vec[:, 0])
        angle_br = np.linspace(-np.pi, np.pi, num=self.lidar_conf.num_bins, endpoint=False)
        angle_hr = angle_hr[:, np.newaxis]
        angle_br = angle_br[np.newaxis, :]
        angle_hb = angle_normalize(angle_hr - angle_br)

        # distance from hazard center to lidar bin
        dist_hb = dist_hr * np.abs(np.sin(angle_hb))
        hb_intersect = dist_hb <= r
        dist_hb = np.minimum(dist_hb, r)

        half_chord = np.sqrt(np.maximum(0, r ** 2 - dist_hb ** 2))

        in_dist = dist_hr * np.cos(angle_hb) + half_chord
        in_lidar = np.exp(in_dist)

        out_dist = dist_hr * np.cos(angle_hb) - half_chord
        out_lidar = hb_intersect * (out_dist > 0) * np.exp(-out_dist)

        in_hazards = dist_hr < r
        lidar = in_hazards * in_lidar + ~in_hazards * out_lidar

        return np.max(lidar, axis=0)


def angle_normalize(x):
    return ((x + math.pi) % (2 * math.pi)) - math.pi
