import numpy as np
from lanro.robots import PyBulletRobot
from lanro.simulation import PyBulletSimulation
from lanro.tasks.core import LanguageTask
from lanro.tasks.scene import basic_scene


class NLLift(LanguageTask):

    def __init__(self,
                 sim: PyBulletSimulation,
                 robot: PyBulletRobot,
                 obj_xy_range: float = 0.3,
                 num_obj: int = 2,
                 min_goal_height: float = 0.0,
                 max_goal_height: float = 0.1,
                 use_hindsight_instructions: bool = False,
                 mode: str = 'Color'):
        super().__init__(sim, robot, mode, use_hindsight_instructions, num_obj)
        self.max_goal_height = max_goal_height
        self.min_goal_height = min_goal_height
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        self.action_verbs = ["lift", "raise", "hoist"]
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer()

    def _create_scene(self) -> None:
        basic_scene(self.sim)

    def reset(self) -> None:
        self.sample_task_objects()
        for idx, obj_pos in zip(self.obj_indices_selection, self._sample_objects()):
            self.sim.set_base_pose(f"object{idx}", obj_pos, [0, 0, 0, 1])
        self._sample_goal()
        self.ep_height_threshold = self.np_random.uniform(low=self.min_goal_height, high=self.max_goal_height)
        # similar to the pick and place task, the goal height is 0 at least 30% of the time
        if self.np_random.random() < 0.3:
            self.ep_height_threshold = 0
        self.reset_hi()

    def grasped_and_lifted(self, obj_body_key):
        obj_pos = np.array(self.sim.get_base_position(obj_body_key))
        hit_obj_id = self.robot.gripper_ray_obs()[0]
        obj_id = self.sim.get_object_id(obj_body_key)
        all_fingers_have_contact = np.all(self.get_contact_with_fingers(obj_body_key))
        achieved_min_height = obj_pos[-1] > self.ep_height_threshold
        inside_gripper = hit_obj_id == obj_id
        return all_fingers_have_contact and achieved_min_height and inside_gripper

    def is_success(self):
        return self.grasped_and_lifted(self.goal_object_body_key)

    def compute_reward(self) -> float:
        if self.is_success():
            return 0.0
        elif self.ep_hindsight_instruction and not self.ep_hindsight_instruction_returned:
            for other_object_idx in self.non_goal_body_indices:
                # if grasped with both fingers and being at a certain height
                if self.grasped_and_lifted(f"object{other_object_idx}"):
                    self.generate_hindsight_instruction(other_object_idx)
                    return -10.
        return -1.0
