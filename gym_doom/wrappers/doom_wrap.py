import cv2
import gym
import numpy as np
from gym import spaces
from gym_doom.config import Config
from gym_doom.doom_env import DoomFuncWrapper


class ViZDoomEnv(DoomFuncWrapper):
    """
        Main wrapper for VizDoom. Change level and set up data directory
    """
    def __init__(self,
                 env,
                 level='deathmatch',
                 data_dir=None):
        """
        :param env: or env_id
        :param level: name of level without extension
        :param data_dir: path to scenarios directory
        """

        Config.init(level, project_dir=data_dir)
        if isinstance(env, str):
            env = gym.make(env)
        super(ViZDoomEnv, self).__init__(env)

        self.viewer = None
        self.setGame(level)

        self.observation_space = self.unwrapped.observation_space
        self.action_space = self.unwrapped.action_space

        # debug
        screen_height = self.unwrapped.game.get_screen_height()
        screen_width = self.unwrapped.game.get_screen_width()
        print("%dX%d" % (screen_height, screen_width))

    def step(self, action):
        if isinstance(action, int):
            action = self.getPossibleActionsCodes()[action]

        elif isinstance(action, (np.ndarray, np.generic)):
            try:
                if action.shape == np.shape([1, ]):
                    action = self.getPossibleActionsCodes()[int(action[0])]
                else:
                    action = action.tolist()
            except AttributeError:
                pass

        return self.env.step(action)

    def render(self, mode='human', **kwargs):
        DoomFuncWrapper.render(self, mode, **kwargs)

    def reset(self):
        return self.env.reset()

    def close(self):
        return self.env.close()


class ScreenWrapper(gym.ObservationWrapper, DoomFuncWrapper):
    def __init__(self, env, dim=None, resolution=None, render=True, dummy_obs=False):
        """
        Screen wrapper for VizDoom.
        Used for resize screen by passing a new dimension tuple to dim.
        Resolution can be changed from default. Must be a string, f.ex. "160x120"
        If you don't want to visualize the main game window set render=False
        In some cases you might check if Reinforcement Learning is actually using your observation space. Then
        set dummy_obs=True and it will return all zeros in observations.

        :param env: VizDoom environment or env_id
        :param dim: resize dimensions as a tuple
        :param resolution: desired resolution
        :param render: renders the main window
        :param dummy_obs: fills observations with zeros
        """
        if isinstance(env, str):
            env = gym.make(env)
        super(ScreenWrapper, self).__init__(env)

        self.setGame(resolution=resolution, render=render)
        self._obs_type = self.doom_env.obs_type
        self.resize_dim = dim
        self.dummy_obs = dummy_obs

        self.screen_resized = self.doom_env.observation_space
        if dim is not None:
            self.screen_resized = spaces.Box(low=0, high=255, shape=(dim[0], dim[1], 3), dtype=np.uint8)

        if self.unwrapped.obs_type == 'image':
            self.observation_space = self.screen_resized

        self._obs = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)

    def observation(self, observation):
        if self.dummy_obs:
            return self._obs
        return self.get_obs()

    def render(self, mode='human', **kwargs):
        DoomFuncWrapper.render(self, mode, **kwargs)

    def resize_screen(self, img):
        if img.shape == self.screen_resized.shape:
            return img
        height = self.screen_resized.shape[0]
        width = self.screen_resized.shape[1]
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def get_image(self):
        return self.resize_screen(self.doom_env.get_image())

    def get_obs(self):
        if self._obs_type == 'ram':
            return self.doom_env.get_ram()
        elif self._obs_type == 'image':
            return self.resize_screen(self.doom_env.get_image())


__all__ = [ViZDoomEnv.__name__, ScreenWrapper.__name__, DoomFuncWrapper.__name__]
