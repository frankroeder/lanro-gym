from lanro.robots import PyBulletRobot
import gym
import numpy as np
from lanro.tasks.core import LanguageTask
from lanro.simulation import PyBulletSimulation
from lanro.utils import goal_distance


class NLReach(LanguageTask):

    def __init__(self,
                 sim: PyBulletSimulation,
                 robot: PyBulletRobot,
                 obj_xy_range: float = 0.3,
                 num_obj: int = 2,
                 use_hindsight_instructions: bool = False,
                 mode: str = 'Color'):
        super().__init__(sim, robot, mode, use_hindsight_instructions, num_obj)
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        self.action_verbs = ["touch", "reach", "contact"]
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer()

    def reset(self) -> None:
        self.sample_task_objects()
        self.obj_init_pos = self._sample_objects()
        for idx, obj_pos in zip(self.obj_indices_selection, self.obj_init_pos):
            self.sim.set_base_pose(f"object{idx}", obj_pos.tolist(), [0, 0, 0, 1])
        self._sample_goal()
        self.reset_hi()

    def is_success(self):
        # NOTE: objects should stay in place with maximum positional change \eta to initial position
        current_obj_pos = np.concatenate(
            [np.array(self.sim.get_base_position(f"object{idx}")) for idx in self.obj_indices_selection])
        close_to_init_pos = goal_distance(np.concatenate(self.obj_init_pos), current_obj_pos) < 0.025
        # check if ticked correct object
        return np.any(self.get_contact_with_fingers(self.goal_object_body_key)) and close_to_init_pos

    def compute_reward(self) -> float:
        if self.is_success():
            return 0.0
        elif self.ep_hindsight_instruction and not self.ep_hindsight_instruction_returned:
            for other_object_idx in self.non_goal_body_indices:
                _non_goal_body = f"object{other_object_idx}"
                # if touched with at least one finger
                if np.any(self.get_contact_with_fingers(_non_goal_body)):
                    self.generate_hindsight_instruction(other_object_idx)
                    return -10.
        return -1.0
