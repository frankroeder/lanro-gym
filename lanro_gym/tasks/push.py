import numpy as np
from lanro_gym.tasks.core import Task
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.tasks.scene import basic_scene
from lanro_gym.env_utils import RGBCOLORS


class Push(Task):

    def __init__(
        self,
        sim: PyBulletSimulation,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        obj_xy_range: float = 0.3,
    ):
        self.sim = sim
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer()

    def _create_scene(self) -> None:
        basic_scene(self.sim)

        self.sim.create_box(
            body_name="object",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=2.0,
            position=[0.0, 0.0, self.object_size / 2],
            rgba_color=RGBCOLORS.RED.value[0] + [1],
        )
        self.sim.create_box(
            body_name="target",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=0.0,
            ghost=True,
            position=[0.0, 0.0, self.object_size / 2],
            rgba_color=RGBCOLORS.RED.value[0] + [0.3],
        )

    def get_obs(self) -> np.ndarray:
        obj_key = "object"
        # position, rotation of the object
        object_position = np.array(self.sim.get_base_position(obj_key))
        object_rotation = np.array(self.sim.get_base_rotation(obj_key))
        object_velocity = np.array(self.sim.get_base_velocity(obj_key))
        object_angular_velocity = np.array(self.sim.get_base_angular_velocity(obj_key))
        observation = np.concatenate([
            object_position,
            object_rotation,
            object_velocity,
            object_angular_velocity,
        ])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position.copy()

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal.tolist(), [0, 0, 0, 1])
        self.sim.set_base_pose("object", object_position.tolist(), [0, 0, 0, 1])

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = [0.0, 0.0, self.object_size / 2] # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal += noise
        return goal.copy()

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = [0.0, 0.0, self.object_size / 2]
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position.copy()
