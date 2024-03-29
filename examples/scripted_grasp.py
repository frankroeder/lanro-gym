"""A script to demonstrate grasping."""
import gymnasium as gym
import lanro_gym
import time as time
import numpy as np

env = gym.make("PandaNLLift2Shape-v0", render=True)
total_ep = 100
start_t = time.time()

for _ in range(total_ep):
    obs, info = env.reset()
    goal_pos = env.sim.get_base_position(env.task.goal_object_body_key)

    for i in range(env._max_episode_steps * 2):
        ee_pos = obs['observation'][:3]
        if i < 35:
            action = np.concatenate((goal_pos - ee_pos, [0]))
            action[3] = 1
        elif i < 45:
            action = np.concatenate((goal_pos - ee_pos, [0]))
            action[3] = -1
        elif i < 60:
            action = np.zeros((4, ))
            action[2] = 0.05
            action[3] = -1

        env.step(action)
