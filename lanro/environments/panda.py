from lanro.robotenv import RobotEnv
from lanro.simulation import PyBulletSimulation
from lanro.robots import Panda
from lanro.tasks import Reach, Push, Stack, Slide


class PandaReachEnv(RobotEnv):

    def __init__(self, render=False, reward_type="sparse", action_type='end_effector'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=True, action_type=action_type)
        task = Reach(
            sim,
            reward_type=reward_type,
            get_ee_position=robot.get_ee_position,
        )
        RobotEnv.__init__(self, sim, robot, task)


class PandaPushEnv(RobotEnv):

    def __init__(self, render=False, reward_type="sparse", action_type='end_effector'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=True, action_type=action_type)
        task = Push(sim, reward_type=reward_type)
        RobotEnv.__init__(self, sim, robot, task)


class PandaSlideEnv(RobotEnv):

    def __init__(self, render=False, reward_type="sparse", action_type='end_effector'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=True, action_type=action_type)
        task = Slide(sim, reward_type=reward_type)
        RobotEnv.__init__(self, sim, robot, task)


class PandaStackEnv(RobotEnv):

    def __init__(self, render=False, reward_type="sparse", num_obj=2, goal_z_range=0.0, action_type='end_effector'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=False, action_type=action_type)
        task = Stack(sim, reward_type=reward_type, num_obj=num_obj, goal_z_range=goal_z_range)
        RobotEnv.__init__(self, sim, robot, task)
