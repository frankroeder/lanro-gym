import os
from typing import Dict, List, Set, Tuple, Union

import gym
from gym import spaces
import numpy as np

from lanro_gym.language_utils import Vocabulary, parse_instructions
from lanro_gym.robots import PyBulletRobot
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.tasks.core import LanguageTask, Task

gym.logger.set_level(40)

DEBUG = int("DEBUG" in os.environ and os.environ["DEBUG"])


class BaseEnv(gym.GoalEnv):
    """
    BaseEnv is a goal-conditoned Gym environment that inherits from `GoalEnv`.
    """
    obs_low = -200.0
    obs_high = 200.0
    ep_counter: int = 0

    def __init__(self,
                 sim: PyBulletSimulation,
                 robot: PyBulletRobot,
                 task: Union[Task, LanguageTask],
                 obs_type: str = "state",
                 seed=None):
        self.sim = sim
        self.metadata = {"render.modes": ["human", "rgb_array"], 'video.frames_per_second': int(np.round(1 / sim.dt))}
        self.robot = robot
        self.action_space = self.robot.action_space
        self.task = task
        self.obs_type = obs_type
        self.seed(seed)

    def seed(self, seed=None) -> List[int]:
        return self.task.seed(seed)

    def close(self) -> None:
        self.sim.close()

    def _get_obs(self):
        raise NotImplementedError

    def getKeyboardEvents(self) -> Dict[int, int]:
        return self.sim.bclient.getKeyboardEvents()

    def reset(self) -> Dict[str, np.ndarray]:
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        self.ep_counter += 1
        return self._get_obs()

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.task.compute_reward(achieved_goal, desired_goal, info)

    def render(self, mode="human"):
        return self.sim.render(mode)


class RobotEnv(BaseEnv):
    ep_end_goal_distance: List = []

    def __init__(self, sim: PyBulletSimulation, robot: PyBulletRobot, task: Task, obs_type: str = "state", seed=None):
        BaseEnv.__init__(self, sim, robot, task, obs_type, seed)

        obs = self.reset()
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(low=self.obs_low,
                                       high=self.obs_high,
                                       shape=obs["observation"].shape,
                                       dtype=np.float32),
                desired_goal=spaces.Box(low=self.obs_low,
                                        high=self.obs_high,
                                        shape=obs["desired_goal"].shape,
                                        dtype=np.float32),
                achieved_goal=spaces.Box(low=self.obs_low,
                                         high=self.obs_high,
                                         shape=obs["achieved_goal"].shape,
                                         dtype=np.float32),
            ))

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs()
        task_obs = self.task.get_obs()
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = self.task.get_achieved_goal()
        desired_goal = self.task.get_goal()
        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": desired_goal.copy(),
        }

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.robot.set_action(action)
        self.sim.step()
        obs = self._get_obs()
        desired_goal = self.task.get_goal()
        done = False
        info = {
            "is_success": self.task.is_success(obs["achieved_goal"], desired_goal),
        }
        reward = self.compute_reward(obs["achieved_goal"], desired_goal, info)

        return obs, reward, done, info

    def reset(self) -> Dict[str, np.ndarray]:
        self.ep_end_goal_distance.append(self.task.last_distance)
        return super().reset()

    def get_metrics(self) -> Dict:
        return {"ep_ctr": self.ep_counter, "avg_terminal_goal_distance": round(np.mean(self.ep_end_goal_distance), 3)}


class RobotLanguageEnv(BaseEnv):
    """
    RobotLanguageEnv is a language-conditioned implementation of `BaseEnv` with a language-specific API.
    """
    discovered_word_idxs: Set = set()

    def __init__(self,
                 sim: PyBulletSimulation,
                 robot: PyBulletRobot,
                 task: LanguageTask,
                 obs_type: str = "state",
                 seed=None):
        BaseEnv.__init__(self, sim, robot, task, obs_type, seed)

        instruction_list = self.task.get_all_instructions()
        if DEBUG:
            print("AMOUNT OF INSTRUCTIONS", len(instruction_list))
        self.word_list, self.max_instruction_len = parse_instructions(instruction_list)
        self.vocab = Vocabulary(self.word_list)
        obs = self.reset()
        self.compute_reward = self.task.compute_reward

        instruction_index_space = spaces.Box(0, len(self.vocab), shape=(self.max_instruction_len, ), dtype=np.uint16)
        if self.obs_type == "state":
            self.observation_space = spaces.Dict(
                dict(
                    observation=spaces.Box(low=self.obs_low,
                                           high=self.obs_high,
                                           shape=obs["observation"].shape,
                                           dtype=np.float32),
                    instruction=instruction_index_space,
                ))
        elif self.obs_type == "pixel":
            self.observation_space = spaces.Dict(
                dict(
                    observation=spaces.Box(low=0, high=255, shape=obs['observation'].shape, dtype=np.uint8),
                    instruction=instruction_index_space,
                ))

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs()
        task_obs = self.task.get_obs()

        if self.obs_type == "pixel":
            observation = self.robot.get_camera_img().copy()
        else:
            observation = np.concatenate([robot_obs, task_obs])
            if self.sim.render_on:
                _ = self.robot.get_camera_img()

        current_goal_string = self.task.get_goal()
        word_representation = self.encode_instruction(self.pad_instruction(current_goal_string))

        return {"observation": observation.copy(), "instruction": word_representation}

    def pad_instruction(self, goal_string):
        _pad_diff = self.max_instruction_len - len(goal_string.split(' '))
        if _pad_diff:
            goal_string += ' ' + ' '.join(['<pad>'] * _pad_diff)
        return goal_string

    def get_vocab(self) -> Vocabulary:
        return self.vocab

    def get_max_instruction_len(self) -> int:
        return self.max_instruction_len

    def encode_instruction(self, instruction) -> np.ndarray:
        word_indices = [self.vocab.word_to_idx(word) for word in instruction.split(' ')]
        return np.array(word_indices)

    def decode_instruction(self, instruction_embedding) -> str:
        words = [self.vocab.idx_to_word(idx) for idx in instruction_embedding]
        return ' '.join(words)

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.robot.set_action(action)
        self.sim.step()
        obs = self._get_obs()
        self.discovered_word_idxs.update(obs['instruction'])
        info = {
            "is_success": self.task.is_success(),
        }
        reward = self.compute_reward()
        done = False

        # HI created, add to info dict and mark episode as done
        if reward == -10.0:
            done = True
            h_instr = self.pad_instruction(self.task.hindsight_instruction)
            info['hindsight_instruction_language'] = h_instr
            info['hindsight_instruction'] = self.encode_instruction(h_instr)
            self.discovered_word_idxs.update(info['hindsight_instruction'])
            # set reward to normal punishment, as we like, e.g., NLReach
            # and NLReachHI to behave the same way
            reward = -1.0

        return obs, reward, done, info

    def get_metrics(self):
        """ Returns a dict of env metrics"""
        return {
            "vocab_discovery_rate": round(len(self.discovered_word_idxs) / (len(self.vocab) - 1), 2),
            "ep_ctr": self.ep_counter,
            **self.task.get_task_metrics(),
        }
