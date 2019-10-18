import random
import time

import gym

from gym_doom.doom_env import DoomFuncWrapper
from gym_doom.wrappers import ViZDoomEnv, ScreenWrapper


def randomAction(env):
    actions = env.unwrapped.available_action_codes
    return random.choice(actions)


if __name__ == '__main__':

    viz_doom_renderer = False
    dir_ = "VizDOOM"
    env = gym.make('Doom-ram-v0')
    env = DoomFuncWrapper(env)
    env = ViZDoomEnv(env, level='deadly_corridor', data_dir=dir_)
    env = ScreenWrapper(env, dim=(100, 100), render=True, dummy_obs=True)

    # env.playHuman()

    obs = env.reset()

    while True:
        action = randomAction(env)

        ob, reward, done, info = env.step(action)

        if not viz_doom_renderer:
            env.render()
        time.sleep(0.01)
        if done:
            env.reset()

    env.close()
