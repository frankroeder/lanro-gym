"""A script to measure the FPS"""
import gymnasium as gym
import lanro_gym
import time as time

env = gym.make("PandaNLPush2HIAR-v0")
# env = gym.make("PandaNLPush2PixelEgoHIAR-v0")
total_ep = 100
step_ctr = 0
start_t = time.time()
for _ in range(total_ep):
    env.reset()
    for _ in range(env._max_episode_steps):
        _ = env.step(env.action_space.sample())
        step_ctr += 1

print(f"FPS: {int(step_ctr / (time.time() - start_t))}")
