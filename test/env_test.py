import gymnasium as gym
import numpy as np
import lanro_gym


def run_random_policy(env):
    env.reset()
    for _ in range(env._max_episode_steps):
        obs = env.step(env.action_space.sample())[0]
        assert np.all(obs['observation'] <= env.observation_space['observation'].high) == True
        assert np.all(obs['observation'] >= env.observation_space['observation'].low) == True
    env.close()


def check_calc_reward(env):
    """Test `compute_reward()` for HER compatibility.""" ""
    obs, _ = env.reset()
    ag = obs['achieved_goal']
    g = obs['desired_goal']
    single_reward = env.compute_reward(ag, g, None)
    assert single_reward in [-1.0, 0.0]
    batch_size = 128
    ag_batch = np.stack([ag for _ in range(batch_size)])
    g_batch = np.stack([g for _ in range(batch_size)])
    batch_reward = env.compute_reward(ag_batch, g_batch, None)
    assert batch_reward.shape[0] == batch_size


def test_env_open_close():
    env = gym.make("PandaReach-v0")
    env.reset()
    env.close()


def test_goal_conditioned_envs():
    render_mode = False
    action_types = [
        'absolute_quat', 'relative_quat', 'relative_joints', 'absolute_joints', 'absolute_rpy', 'relative_rpy',
        'end_effector'
    ]
    for a_type in action_types:
        for robot in ['Panda']:
            for task in ['Reach', 'Push', 'Slide', 'PickAndPlace']:
                run_random_policy(gym.make(f'{robot}{task}-v0', render=render_mode, action_type=a_type))
                check_calc_reward(gym.make(f'{robot}{task}-v0'))
            for task in ['Stack2', 'Stack3', 'Stack4']:
                run_random_policy(gym.make(f'{robot}{task}-v0', render=render_mode, action_type=a_type))
                check_calc_reward(gym.make(f'{robot}{task}-v0'))


def test_language_conditioned_envs():
    render_mode = False
    obj_count = 2
    action_types = [
        'absolute_quat', 'relative_quat', 'relative_joints', 'absolute_joints', 'absolute_rpy', 'relative_rpy',
        'end_effector'
    ]
    for a_type in action_types:
        for robot in ['Panda']:
            for lang_task in ['NLReach', 'NLPush', 'NLGrasp', 'NLLift']:
                for _mode in ['', 'Color', 'Shape', 'Weight', 'Size', 'ColorShape', 'WeightShape', 'SizeShape']:
                    for _obstype in ["", "PixelEgo", "PixelStatic"]:
                        for _use_syn in ["", "Synonyms"]:
                            for _hindsight_instr in ["", "HI"]:
                                for _action_repair in ["", "AR", "ARN", "ARD", "ARND"]:
                                    id = f'{robot}{lang_task}{obj_count}{_mode}{_obstype}{_use_syn}{_hindsight_instr}{_action_repair}-v0'
                                    run_random_policy(gym.make(id, render=render_mode, action_type=a_type))
