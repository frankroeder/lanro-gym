import itertools
from typing import Dict, List

from gym import utils as gym_utils
import numpy as np

from lanro.robots.pybrobot import PyBulletRobot
from lanro.simulation import PyBulletSimulation
from lanro.tasks.scene import basic_scene
import lanro.utils as play_utils
from lanro.utils import TaskObjectList
from lanro.language_utils import create_commands


class Task:
    distance_threshold = 0.05
    reward_type: str = "sparse"
    last_distance: float = 0
    object_size: float = 0.04

    def get_goal(self):
        """Return the current goal."""
        raise NotImplementedError

    def get_obs(self):
        """Return the observation associated to the task."""
        raise NotImplementedError

    def get_achieved_goal(self):
        """Return the achieved goal."""
        raise NotImplementedError

    def reset(self):
        """Reset the task"""
        raise NotImplementedError

    def seed(self, seed):
        self.np_random, seed = gym_utils.seeding.np_random(seed)
        return [seed]

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        distance = play_utils.goal_distance(achieved_goal, desired_goal)
        return float(distance < self.distance_threshold)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> float:
        distance = play_utils.goal_distance(achieved_goal, desired_goal)
        self.last_distance = distance
        if self.reward_type == "sparse":
            return -(distance > self.distance_threshold).astype(np.float32)
        else:
            return -distance

    def get_task_metrics(self) -> Dict:
        return {}


class LanguageTask:
    distance_threshold: float = 0.05
    object_size: float = 0.04
    current_instruction: np.ndarray
    action_verbs: List = []
    secondary_props_words: List = []
    obj_range_low: np.ndarray
    obj_range_high: np.ndarray
    task_object_list: TaskObjectList
    num_obj: int = 0

    # hindsight instruction flags and metrics
    hindsight_instruction = None
    use_hindsight_instructions: bool = False
    total_hindsight_instruction_episodes: int = 0
    discovered_hindsight_instruction_ctr: int = 0

    # pybullet user debug text id
    instruction_sim_id = 43

    def __init__(self, sim: PyBulletSimulation, robot: PyBulletRobot, mode: str, use_hindsight_instructions: bool,
                 num_obj: int):
        self.sim = sim
        self.robot = robot
        self.use_hindsight_instructions = use_hindsight_instructions
        self.num_obj = num_obj
        self.mode = mode

        _args = dict(color_mode='color' in mode)
        if 'shape' in mode:
            self.task_object_list = TaskObjectList(sim, shape_mode=True, **_args)
        else:
            self.task_object_list = TaskObjectList(sim, **_args)

        self.object_properties = self.task_object_list.get_obj_properties()

    def get_goal(self) -> str:
        """ Get the goal as instruction string """
        return self.current_instruction.item()

    def reset(self):
        """Reset the task: sample a new goal"""
        raise NotImplementedError

    def is_success(self) -> float:
        raise NotImplementedError

    def compute_reward(self) -> float:
        raise NotImplementedError

    def seed(self, seed):
        """Sets the seed for this env's random number."""
        self.np_random, seed = gym_utils.seeding.np_random(seed)
        return [seed]

    def _create_scene(self) -> None:
        basic_scene(self.sim)

    def get_all_instructions(self):
        instruction_set = np.concatenate(
            [create_commands(self.action_verbs, _property_tuple) for _property_tuple in self.object_properties])
        return list(set(instruction_set))

    def get_obs(self) -> np.ndarray:
        observation = []
        for idx in self.obj_indices_selection:
            obj_key = f"object{idx}"
            object_position = np.array(self.sim.get_base_position(obj_key))
            object_rotation = np.array(self.sim.get_base_rotation(obj_key))
            object_velocity = np.array(self.sim.get_base_velocity(obj_key))
            object_angular_velocity = np.array(self.sim.get_base_angular_velocity(obj_key))
            object_identifier = self.task_object_list.objects[idx].get_onehot()
            observation.extend(
                [object_position, object_rotation, object_velocity, object_angular_velocity, object_identifier])
        return np.concatenate(observation)

    def get_contact_with_fingers(self, target_body) -> List:
        # check contact with fingers defined by ee_joints
        finger_contacts = [
            bool(self.sim.get_contact_points(target_body, self.robot.body_name, linkIndexB=finger_idx))
            # assume the first two indices are the fingers of the end effector
            for finger_idx in self.robot.ee_joints[:2]
        ]
        return finger_contacts

    def _sample_goal(self) -> None:
        """Randomly select one of the generated instructions for the current goal object"""
        property_tuple = self.task_object_list.objects[self.goal_obj_idx].get_properties()
        sentences = create_commands(self.action_verbs, property_tuple)
        self.current_instruction = self.np_random.choice(sentences, 1)
        self.sim.bclient.addUserDebugText(self.get_goal(), [0.05, -.3, .4],
                                          textSize=2.0,
                                          replaceItemUniqueId=self.instruction_sim_id)

    def _sample_objects(self):
        """Randomize start position of objects."""
        while True:
            obj_positions = [[0.0, 0.0, self.object_size / 2] +
                             self.np_random.uniform(self.obj_range_low, self.obj_range_high)
                             for _ in range(self.num_obj)]
            unique_distance_combinations = [np.linalg.norm(a - b) for a, b in itertools.combinations(obj_positions, 2)]
            # if minimal distance between two objects is greater than three times
            # the object size (as objects should not be on top of each other
            # and we like to have a minimal distance between them)
            if np.min(unique_distance_combinations) > self.object_size * 3:
                return tuple(obj_positions)

    def sample_task_objects(self):
        # remove old objects, as we do not want to destroy the whole simulation
        remove_obj_keys = [key for key in self.sim._bodies_idx.keys() if 'object' in key]
        for _key in remove_obj_keys:
            self.sim.remove_body(_key)

        # TODO: Check and ensure we only have duplicates along one feature dimension (color xor shape) #
        # select object indices from the list of all possible task objects
        self.obj_indices_selection = self.np_random.choice(len(self.task_object_list), size=self.num_obj, replace=False)
        self.goal_obj_idx = self.np_random.choice(self.obj_indices_selection, 1)[0]
        self.non_goal_body_indices = [idx for idx in self.obj_indices_selection if idx != self.goal_obj_idx]
        self.goal_object_body_key = f"object{self.goal_obj_idx}"

        for obj_idx in self.obj_indices_selection:
            object_body_key = f"object{obj_idx}"
            self.task_object_list.objects[obj_idx].load(object_body_key)

    def generate_hindsight_instruction(self, _obj_idx):
        property_tuple = self.task_object_list.objects[_obj_idx].get_properties()
        hindsight_sentences = create_commands(self.action_verbs, property_tuple)
        self.discovered_hindsight_instruction_ctr += 1
        self.ep_hindsight_instruction_returned = True
        self.hindsight_instruction = self.np_random.choice(hindsight_sentences, 1)[0]

    def reset_hi(self):
        self.ep_hindsight_instruction = False
        self.ep_hindsight_instruction_returned = False

        if self.use_hindsight_instructions:
            # generate hindsight instructions 25% the time
            self.ep_hindsight_instruction = self.np_random.random() < 0.25

        if self.ep_hindsight_instruction:
            self.total_hindsight_instruction_episodes += 1

    def get_task_metrics(self) -> Dict:
        """ Returns a dict of task-specific metrics """
        metrics = {}
        if self.use_hindsight_instructions:
            metrics["HI_episodes"] = self.total_hindsight_instruction_episodes
            if self.total_hindsight_instruction_episodes:
                metrics["HI_discovery_rate"] = round(
                    self.discovered_hindsight_instruction_ctr / self.total_hindsight_instruction_episodes, 2)
        return metrics
