import numpy as np
import gymnasium as gym
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


def test_single_hi_env():
    env = gym.make("PandaNLReach2HI-v0", render=False)
    env.reset()
    assert env.task.use_hindsight_instructions == True
    assert env.task.use_action_repair == False
    env.task.generate_hindsight_instruction(1)
    assert len(env.task.hindsight_instruction)
    assert len(env.task.get_all_instructions()) == 9


def test_single_hi_env_synonyms():
    env = gym.make("PandaNLReach2SynonymsHI-v0", render=False)
    env.reset()
    assert env.task.use_hindsight_instructions == True
    assert env.task.use_action_repair == False
    env.task.generate_hindsight_instruction(1)
    assert len(env.task.hindsight_instruction)
    assert len(env.task.get_all_instructions()) == 18


def test_single_ar_env():
    env = gym.make("PandaNLReach2AR-v0", render=False)
    env.reset()
    assert env.task.use_hindsight_instructions == False
    assert env.task.use_action_repair == True
    assert len(env.task.get_all_instructions()) == 171


def test_single_ar_env_synonyms():
    env = gym.make("PandaNLReach2SynonymsAR-v0", render=False)
    env.reset()
    assert env.task.use_hindsight_instructions == False
    assert env.task.use_action_repair == True
    assert len(env.task.get_all_instructions()) == 666


def test_single_arn_env():
    env = gym.make("PandaNLReach2ARN-v0", render=False)
    env.reset()
    assert env.task.use_hindsight_instructions == False
    assert env.task.use_action_repair == True
    assert len(env.task.get_all_instructions()) == 198


def test_single_arn_env_synonyms():
    env = gym.make("PandaNLReach2SynonymsARN-v0", render=False)
    env.reset()
    assert env.task.use_hindsight_instructions == False
    assert env.task.use_action_repair == True
    assert len(env.task.get_all_instructions()) == 774


def test_nl_envs():
    for robot in ['Panda']:
        for lang_task in ['NLReach', 'NLPush', 'NLGrasp', 'NLLift']:
            for obj_count in [2]:
                for _mode in [
                        '', 'Color', 'Shape', 'Weight', 'Size', 'ColorShape', 'WeightShape', 'SizeShape',
                        'ColorShapeSize'
                ]:
                    for _obstype in ["", "PixelEgo", "PixelStatic"]:
                        for _use_syn in ["", "Synonyms"]:
                            for _hindsight_instr in ["", "HI"]:
                                for _action_repair in ["", "AR", "ARN", "ARD", "ARND"]:
                                    id = f'{robot}{lang_task}{obj_count}{_mode}{_obstype}{_use_syn}{_hindsight_instr}{_action_repair}-v0'
                                    env = gym.make(id, render=False)
                                    obs, _ = env.reset()
                                    check_instruction(env, obs)
                                    env.close()


def test_pixel_envs():
    for lang_task in ['NLReach', 'NLPush', 'NLGrasp', 'NLLift']:
        for _pixel_obstype in ["PixelEgo", "PixelStatic"]:
            env = gym.make(f"Panda{lang_task}2{_pixel_obstype}-v0")
            obs, _ = env.reset()
            img = obs['observation']
            assert img.shape == (84, 84, 3)
            assert env.observation_space['observation'].shape == (84, 84, 3)
            assert env.observation_space['observation'].dtype == np.uint8
            env.close()
