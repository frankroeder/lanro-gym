import gym
import os
import numpy as np
import lanro
import argparse
import glfw

DEBUG = int("DEBUG" in os.environ and os.environ["DEBUG"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--interactive', action='store_true', dest='interactive', help='Start interactive mode')
    parser.add_argument('-t', '--test', action='store_true', dest='test', help='Start test mode')
    parser.add_argument('-r', '--reward', action='store_true', dest='reward', help='Print the reward.')
    parser.add_argument('-a', '--action', action='store_true', dest='action', help='Print the action.')
    parser.add_argument('--full', action='store_true', dest='full', help='Print everything')
    parser.add_argument('--keyboard',
                        action='store_true',
                        dest='keyboard_control',
                        help='Activates keyboard control for interactive mode.')
    parser.add_argument('--metrics', action='store_true', help='Print environment metrics.')
    parser.add_argument('--action_type', type=str, default='absolute_joints', help='Action type to control the robot.')
    parser.add_argument(
        '-e',
        '--env',
        default='PandaNLReach2-v0',
        help=
        f"Available envs: {', '.join([envspec.id for envspec in gym.envs.registry.all() if 'Panda' in envspec.id or 'UR5' in envspec.id])}"
    )
    return parser.parse_args()


def log_step(env, action, args):
    obs, reward, done, info = env.step(action)
    if args.reward:
        print(f"reward: {reward} success: {info['is_success']}")
    if args.action:
        print(action)
    if args.full:
        print(obs, reward, done, info)
    if args.metrics:
        print(env.get_metrics())
    if DEBUG and info['is_success'] or 'hindsight_instruction' in info.keys():
        import ipdb
        ipdb.set_trace()
    return done or info['is_success']


def test(env, args):
    for _ in range(100):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            done = log_step(env, action, args)
            env.render(mode="human")


key_events = {
    65297: "forward",
    65298: "backward",
    65295: "straight_left",
    65296: "straight_right",
    glfw.KEY_MINUS: "close_gripper",
    glfw.KEY_5: "open_gripper",
    43: "open_gripper",
    glfw.KEY_8: "up",
    glfw.KEY_2: "down",
    glfw.KEY_1: "yaw_left",
    glfw.KEY_3: "yaw_right",
    glfw.KEY_6: "pitch_right",
    glfw.KEY_4: "pitch_left",
    glfw.KEY_7: "roll_left",
    glfw.KEY_9: "roll_right",
}


def interactive(args):
    env = gym.make(args.env, render=True, action_type=args.action_type)

    if not args.keyboard_control:
        controls = env.robot.get_xyz_rpy_controls()

    for _ in range(10):
        env.reset()
        done = False
        action = np.zeros(shape=env.action_space.shape)
        key_control_gain = 0.01
        for idx, val in enumerate(env.robot.get_default_controls().values()):
            if len(action) > idx:
                action[idx] = val

        while True:
            if args.keyboard_control:
                keys = env.getKeyboardEvents()
                if keys:
                    key_str = ''.join(
                        [key_events[_pressed] for _pressed in keys.keys() if _pressed in key_events.keys()])
                    if "forward" in key_str:
                        action[3] += 1 * key_control_gain
                    if "backward" in key_str:
                        action[3] += -1 * key_control_gain
                    if "straight_left" in key_str:
                        action[0] += 1 * key_control_gain
                    if "straight_right" in key_str:
                        action[0] += -1 * key_control_gain
                    if "up" in key_str:
                        action[1] += -1 * key_control_gain
                    if "down" in key_str:
                        action[1] += 1 * key_control_gain
                    if not env.robot.fixed_gripper:
                        if "close_gripper" in key_str:
                            action[-1] += 1 * key_control_gain
                        if "open_gripper" in key_str:
                            action[-1] += -1 * key_control_gain
                    if env.action_space.shape[0] > 4:
                        if "roll_left" in key_str:
                            action[2] += 1 * key_control_gain
                        if "roll_right" in key_str:
                            action[2] += -1 * key_control_gain
                        if "pitch_left" in key_str:
                            action[4] += 1 * key_control_gain
                        if "pitch_right" in key_str:
                            action[4] += -1 * key_control_gain
                        if "yaw_left" in key_str:
                            action[5] += -1 * key_control_gain
                        if "yaw_right" in key_str:
                            action[5] += 1 * key_control_gain
            else:
                action = np.zeros(shape=env.action_space.shape)
                for idx, ctrl_id in enumerate(controls):
                    try:
                        action[idx] = env.sim.bclient.readUserDebugParameter(ctrl_id)
                    except Exception as e:
                        print(e)
                        continue

            done = log_step(env, np.array(action), args)
            env.render(mode='human')
            if args.metrics and done:
                break


def main():
    args = parse_args()
    if args.test:
        env = gym.make(args.env, render=True)
        env.reset()
        test(env, args)
        env.close()
    elif args.interactive:
        interactive(args)
    else:
        raise ValueError("No valid mode found: use -t/--test (test mode) or -i/--interactive (interactive mode)")


if __name__ == '__main__':
    main()
