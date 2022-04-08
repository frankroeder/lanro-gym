from gym.envs.registration import register

robot = 'Panda'
for reward_type in ["sparse", "dense"]:
    _r_type = "Dense" if reward_type == "dense" else ""
    kwargs = {
        "reward_type": reward_type,
    }
    register(
        id=f'{robot}Reach{_r_type}-v0',
        entry_point='lanro.environments:{}ReachEnv'.format(robot),
        max_episode_steps=50,
        kwargs=kwargs,
    )
    register(
        id=f'{robot}Push{_r_type}-v0',
        entry_point='lanro.environments:{}PushEnv'.format(robot),
        max_episode_steps=50,
        kwargs=kwargs,
    )
    register(
        id=f'{robot}Slide{_r_type}-v0',
        entry_point='lanro.environments:{}SlideEnv'.format(robot),
        max_episode_steps=50,
        kwargs=kwargs,
    )
    register(
        id=f'{robot}PickAndPlace{_r_type}-v0',
        entry_point='lanro.environments:{}StackEnv'.format(robot),
        max_episode_steps=50,
        kwargs={
            **kwargs,
            'num_obj': 1,
            'goal_z_range': 0.2,
        },
    )
    for num_obj in [2, 3, 4]:
        register(
            id=f'{robot}Stack{num_obj}{_r_type}-v0',
            entry_point='lanro.environments:{}StackEnv'.format(robot),
            max_episode_steps=50 * num_obj,
            kwargs={
                **kwargs, 'num_obj': num_obj
            },
        )

for num_obj in [2, 3]:
    for _mode in ['Default', 'Color', 'Shape', 'ColorShape']:
        for _obstype in ['state', 'pixelego', 'pixelstatic']:
            _current_obstype = ''
            _cam_mode = 'ego'
            if _obstype == 'pixelego':
                _current_obstype = 'PixelEgo'
                _obstype = 'pixel'
            elif _obstype == 'pixelstatic':
                _cam_mode = 'static'
                _current_obstype = 'PixelStatic'
                _obstype = 'pixel'
            for _h_instr in [True, False]:
                _current_mode = '' if _mode == 'Default' else _mode
                _current_h_instr = 'HI' if _h_instr else ''
                _max_episode_steps = 50

                _kwargs = {
                    'num_obj': num_obj,
                    'mode': _mode.lower(),
                    'obs_type': _obstype,
                    'use_hindsight_instructions': _h_instr,
                    'camera_mode': _cam_mode,
                }

                param_combination = f"{num_obj}{_current_mode}{_current_obstype}{_current_h_instr}"

                register(id=f'{robot}NLReach{param_combination}-v0',
                         entry_point='lanro.environments:{}NLReachEnv'.format(robot),
                         max_episode_steps=_max_episode_steps,
                         kwargs=_kwargs)
                register(id=f'{robot}NLPush{param_combination}-v0',
                         entry_point='lanro.environments:{}NLPushEnv'.format(robot),
                         max_episode_steps=_max_episode_steps,
                         kwargs=_kwargs)
                register(id=f'{robot}NLGrasp{param_combination}-v0',
                         entry_point='lanro.environments:{}NLGraspEnv'.format(robot),
                         max_episode_steps=_max_episode_steps,
                         kwargs=_kwargs)
                register(id=f'{robot}NLLift{param_combination}-v0',
                         entry_point='lanro.environments:{}NLLiftEnv'.format(robot),
                         max_episode_steps=_max_episode_steps,
                         kwargs=_kwargs)
