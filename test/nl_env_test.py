import numpy as np
import gym
import lanro_gym
from lanro_gym.language_utils import parse_instructions


def check_instruction(env, obs):
    instruction_representation = obs['instruction']
    sentence = env.decode_instruction(instruction_representation)
    instruction_representation2 = env.encode_instruction(sentence)
    assert np.all(instruction_representation == instruction_representation2)
    assert sentence == env.pad_instruction(env.task.current_instruction[0])
    instruction_list = env.task.get_all_instructions()
    word_list, max_instruction_len = parse_instructions(instruction_list)
    instruction_space = env.observation_space['instruction']
    assert instruction_space.high[-1] == len(word_list) + 1 # for <pad> token
    assert instruction_space.shape[0] == max_instruction_len


def test_nl_envs():
    for robot in ['Panda']:
        for lang_task in ['NLReach', 'NLPush', 'NLGrasp', 'NLLift']:
            for obj_count in [2]:
                for _mode in ['', 'Color', 'Shape', 'ColorShape']:
                    for _obstype in ["", "PixelEgo", "PixelStatic"]:
                        for _hindsight_instr in ["", "HI"]:
                            id = f'{robot}{lang_task}{obj_count}{_mode}{_obstype}{_hindsight_instr}-v0'
                            env = gym.make(id, render=False)
                            obs = env.reset()
                            check_instruction(env, obs)
                            env.close()


def test_pixel_envs():
    for lang_task in ['NLReach', 'NLPush', 'NLGrasp', 'NLLift']:
        for _pixel_obstype in ["PixelEgo", "PixelStatic"]:
            env = gym.make(f"Panda{lang_task}2{_pixel_obstype}-v0")
            obs = env.reset()
            img = obs['observation']
            assert img.shape == (84, 84, 3)
            assert env.observation_space['observation'].shape == (84, 84, 3)
            assert env.observation_space['observation'].dtype == np.uint8
            env.close()
