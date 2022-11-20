import numpy as np
from typing import Callable
from lanro_gym.tasks.core import Task
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.tasks.scene import basic_scene
from lanro_gym.env_utils import RGBCOLORS


class Reach(Task):

    def __init__(self,
                 sim: PyBulletSimulation,
                 get_ee_position: Callable[[], np.ndarray],
                 reward_type: str = "sparse",
                 distance_threshold: float = 0.025,
                 goal_range: float = 0.3):
        self.sim = sim
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer()

    def _create_scene(self) -> None:
        basic_scene(self.sim)
        self.sim.create_sphere(
            body_name="target",
            radius=self.distance_threshold,
            mass=0.0,
            ghost=True,
            position=[0.0, 0.0, 0.0],
            rgba_color=RGBCOLORS.RED.value[0] + [0.3],
        )

    def get_obs(self) -> np.ndarray:
        return np.array([])

    def get_achieved_goal(self) -> np.ndarray:
        return self.get_ee_position()

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal.tolist(), [0, 0, 0, 1])

    def _sample_goal(self) -> np.ndarray:
        return self.np_random.uniform(self.goal_range_low, self.goal_range_high)
