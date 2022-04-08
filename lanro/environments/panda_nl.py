from lanro.robotenv import RobotLanguageEnv
from lanro.simulation import PyBulletSimulation
from lanro.robots import Panda
from lanro.tasks import NLReach, NLLift, NLGrasp, NLPush


class PandaNLReachEnv(RobotLanguageEnv):

    def __init__(self,
                 render=False,
                 num_obj=2,
                 obs_type="state",
                 mode='Color',
                 action_type='end_effector',
                 use_hindsight_instructions=False,
                 camera_mode='ego'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=True, action_type=action_type, camera_mode=camera_mode)
        task = NLReach(sim, robot, num_obj=num_obj, mode=mode, use_hindsight_instructions=use_hindsight_instructions)
        RobotLanguageEnv.__init__(self, sim, robot, task, obs_type=obs_type)


class PandaNLPushEnv(RobotLanguageEnv):

    def __init__(self,
                 render=False,
                 num_obj=2,
                 obs_type="state",
                 mode='Color',
                 action_type='end_effector',
                 use_hindsight_instructions=False,
                 camera_mode='ego'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=True, action_type=action_type, camera_mode=camera_mode)
        task = NLPush(sim, robot, num_obj=num_obj, mode=mode, use_hindsight_instructions=use_hindsight_instructions)
        RobotLanguageEnv.__init__(self, sim, robot, task, obs_type=obs_type)


class PandaNLGraspEnv(RobotLanguageEnv):

    def __init__(self,
                 render=False,
                 num_obj=2,
                 obs_type="state",
                 mode='Color',
                 action_type='end_effector',
                 use_hindsight_instructions=False,
                 camera_mode='ego'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=False, action_type=action_type, camera_mode=camera_mode)
        task = NLGrasp(sim, robot, num_obj=num_obj, mode=mode, use_hindsight_instructions=use_hindsight_instructions)
        RobotLanguageEnv.__init__(self, sim, robot, task, obs_type=obs_type)


class PandaNLLiftEnv(RobotLanguageEnv):

    def __init__(self,
                 render=False,
                 num_obj=2,
                 obs_type="state",
                 mode='Color',
                 action_type='end_effector',
                 use_hindsight_instructions=False,
                 camera_mode='ego'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=False, action_type=action_type, camera_mode=camera_mode)
        task = NLLift(sim, robot, num_obj=num_obj, mode=mode, use_hindsight_instructions=use_hindsight_instructions)
        RobotLanguageEnv.__init__(self, sim, robot, task, obs_type=obs_type)
