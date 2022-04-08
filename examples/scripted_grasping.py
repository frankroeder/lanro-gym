import gym
import lanro
import numpy as np

env = gym.make("PandaNLLift3Shape-v0", render=True)
total_ep = 100
for _ in range(total_ep):
    obs = env.reset()
    goal_pos = env.sim.get_base_position(env.task.goal_object_body_key)
    for i in range(env._max_episode_steps * 2):
        ee_pos = obs['observation'][:3]
        if i < 30:
            action = np.concatenate((goal_pos - ee_pos, [0]))
            action[3] = 1
        elif i < 40:
            action = np.concatenate((goal_pos - ee_pos, [0]))
            action[3] = -1
        elif i < 50:
            action = np.zeros((4, ))
            action[2] = 0.05
            action[3] = -1

        next_obs, _, _, _ = env.step(action)
