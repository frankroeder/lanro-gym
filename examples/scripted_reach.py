"""A script to demonstrate reaching."""
import gymnasium as gym
import lanro_gym
import time as time
import numpy as np

env = gym.make("PandaReach-v0", render=True)
total_ep = 100
start_t = time.time()

for _ in range(total_ep):
    obs, info = env.reset()
    goal_pos = obs['desired_goal']

    for i in range(env._max_episode_steps * 4):
        ee_pos = obs['achieved_goal']
        action = (goal_pos - ee_pos).copy()
        if i < 25:
            action *= 0.75
        elif i < 50:
            action *= 0.5
        elif i < 75:
            action *= 0.5
        else:
            action = np.zeros_like(action)
        env.step(action)
