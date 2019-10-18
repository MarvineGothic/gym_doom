import itertools as it
import logging
import multiprocessing
import random
import sys
from time import sleep

import gym
import numpy as np
import tqdm
from gym import spaces, error
from gym.envs.classic_control import rendering
from gym.utils import seeding
from gym_doom.config import *
from gym_doom.utils import action_to_buttons
from stable_baselines.common import EzPickle

try:
    import vizdoom
    from vizdoom import DoomGame, Mode, Button, GameVariable, ScreenFormat, ScreenResolution, GameState  # , Loader
    from vizdoom import ViZDoomUnexpectedExitException, ViZDoomErrorException
except ImportError as e:
    raise gym.error.DependencyNotInstalled("{}. (HINT: you can install Doom dependencies " +
                                           "with 'pip install doom_py.)'".format(e))

logger = logging.getLogger(__name__)

resolutions = ['160x120', '200x125', '200x150', '256x144', '256x160', '256x192', '320x180', '320x200',
               '320x240', '320x256', '400x225', '400x250', '400x300', '512x288', '512x320', '512x384',
               '640x360', '640x400', '640x480', '800x450', '800x500', '800x600', '1024x576', '1024x640',
               '1024x768', '1280x720', '1280x800', '1280x960', '1280x1024', '1400x787', '1400x875',
               '1400x1050', '1600x900', '1600x1000', '1600x1200', '1920x1080']

inverted_screen_formats = [ScreenFormat.CRCGCB]


# Singleton pattern
class DoomLock:
    class __DoomLock:
        def __init__(self):
            self.lock = multiprocessing.Lock()

    instance = None

    def __init__(self):
        if not DoomLock.instance:
            DoomLock.instance = DoomLock.__DoomLock()

    def get_lock(self):
        return DoomLock.instance.lock


class DoomEnv(gym.Env, EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 35}

    def __init__(self, level='deathmatch', obs_type='ram'):
        # super(DoomEnv, self).__init__()
        EzPickle.__init__(self, level.split('.')[0], obs_type)
        assert obs_type in ('ram', 'image')
        level = level.split('.')[0]
        Config.init(level)

        self.curr_seed = 0
        self.game = DoomGame()
        self.lock = (DoomLock()).get_lock()

        self.level = level
        self.obs_type = obs_type
        self.tick = 4

        self._mode = 'algo'

        self.is_render_in_human_mode = True
        self.is_game_initialized = False
        self.is_level_loaded = False

        self.viewer = None

        self.set_game(self.level, resolution=None, render=True)
        print()

    # todo: add frame skip option by using tick
    def step(self, action):
        reward = 0.0
        # self.tick = 4
        if self._mode == 'algo':
            if self.tick:
                reward = self.game.make_action(action, self.tick)
            else:
                reward = self.game.make_action(action)

            # self.game.set_action(action)
            # self.game.advance_action(4)
            # reward = self.game.get_last_reward()

        return self.get_obs(), reward, self.isDone(), self.get_info()

    def reset(self):
        if not self.is_game_initialized:
            self.__load_level()
            self.__init_game()

        self.__start_episode()
        return self.get_obs()

    def render(self, mode='human', **kwargs):
        if 'close' in kwargs and kwargs['close']:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if mode == 'human' and not self.is_render_in_human_mode:
            return
        img = self.get_image()

        if mode == 'rgb_array':
            return img
        elif mode is 'human':
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def close(self):
        with self.lock:
            self.game.close()

    def seed(self, seed=None):
        self.curr_seed = seeding.hash_seed(seed) % 2 ** 32
        return [self.curr_seed]

    # ================================== GETTERS SETTERS ===============================================================
    def set_game(self, level, resolution, render):
        self.__configure()
        self.__load_level(level)
        self.__set_resolution(resolution)
        self.__set_obs_and_ac_space()
        self.__set_player(render)

    def __configure(self, lock=None, **kwargs):
        self.seed()
        if lock is not None:
            self.lock = lock

    def __load_level(self, level=None):
        if level is not None:
            self.level = level.split('.')[0]
            self.is_level_loaded = False

        if self.is_level_loaded:
            return
        if self.is_game_initialized:
            self.is_game_initialized = False
            self.game.close()
            self.game = DoomGame()

        if not self.is_game_initialized:
            self.game.set_vizdoom_path(Config.VIZDOOM_PATH)
            self.game.set_doom_game_path(Config.FREEDOOM_PATH)

        # Common settings
        self.record_file_path = Config.RECORD_FILE_PATH
        self.game.load_config(Config.VIZDOOM_SCENARIO_PATH + Config.DOOM_SETTINGS[self.level][Config.CONFIG])
        self.game.set_doom_scenario_path(
            Config.VIZDOOM_SCENARIO_PATH + Config.DOOM_SETTINGS[self.level][Config.SCENARIO])

        if Config.DOOM_SETTINGS[self.level][Config.MAP] != '':
            self.game.set_doom_map(Config.DOOM_SETTINGS[self.level][Config.MAP])
        self.game.set_doom_skill(Config.DOOM_SETTINGS[self.level][Config.DIFFICULTY])

        self.allowed_actions = Config.DOOM_SETTINGS[self.level][Config.ACTIONS]
        self.available_game_variables = Config.DOOM_SETTINGS[self.level][Config.GAME_VARIABLES]

        self.is_level_loaded = True

    def __set_resolution(self, resolution=None):
        if resolution is None:
            resolution = Config.DEFAULT_SCREEN_RESOLUTION
        resolution_l = resolution.lower()
        if resolution_l not in resolutions:
            raise gym.error.Error(
                'Error - The specified resolution "{}" is not supported by Vizdoom.\n The list of valid'
                'resolutions: {}'.format(resolution, resolutions))
        if '_' in resolution_l:
            resolution_l = resolution_l.split('_')[1]
        self.scr_width = int(resolution_l.split("x")[0])
        self.scr_height = int(resolution_l.split("x")[1])
        self.game.set_screen_resolution(getattr(ScreenResolution, 'RES_{}X{}'.format(self.scr_width, self.scr_height)))

        self.screen_format = self.game.get_screen_format()
        self.screen_height = self.game.get_screen_height()
        self.screen_width = self.game.get_screen_width()

    def __set_obs_and_ac_space(self):
        if self.obs_type == 'ram':
            self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8,
                                                shape=(len(self.available_game_variables),))
        elif self.obs_type == 'image':
            # self.observation_space = self.screen_resized
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.scr_height, self.scr_width, 3),
                                                dtype=np.uint8)
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self.obs_type))

        if self.screen_format in inverted_screen_formats:
            self.dummy_screen = np.zeros(shape=(3, self.scr_height, self.scr_width), dtype=np.uint8)
        else:
            self.dummy_screen = np.zeros(shape=(self.scr_height, self.scr_width, 3), dtype=np.uint8)

        self.dummy_ram = [0] * len(self.available_game_variables)

        self.available_action_codes = [list(a) for a in
                                       it.product([0, 1], repeat=self.game.get_available_buttons_size())]
        # self.__delete_conflict_actions()
        self.action_space = spaces.MultiDiscrete([len(self.available_action_codes)])

    def __set_player(self, render=True):
        self.game.set_window_visible(render)
        self.game.set_mode(Mode.PLAYER)

    def __init_game(self):
        try:
            with self.lock:
                self.game.init()
                self.is_game_initialized = True
        except (ViZDoomUnexpectedExitException, ViZDoomErrorException):
            raise error.Error(
                'Could not start the game.')

    def __start_episode(self):
        if self.curr_seed > 0:
            self.game.set_seed(self.curr_seed)
            self.curr_seed = 0
        if self.record_file_path:
            self.game.new_episode(self.record_file_path)
        else:
            self.game.new_episode()
        return

    def getState(self):
        return self.game.get_state()

    def getLastAction(self):
        return self.game.get_last_action()

    def getButtonsNames(self, action):
        return action_to_buttons(self.allowed_actions, action)

    def get_info(self):
        info = {"LEVEL": self.level,
                "TOTAL_REWARD": round(self.game.get_total_reward(), 4)}

        state_variables = self.get_ram()
        for i in range(len(self.available_game_variables)):
            info[self.available_game_variables[i]] = state_variables[i]

        return info

    def get_ram(self):
        if not self.is_game_initialized:
            raise NotImplementedError("The game was not initialized. Run env.reset() first!")
        try:
            ram = self.getState().game_variables
        except AttributeError:
            ram = self.dummy_ram
        return ram

    def get_image(self):
        try:
            screen = self.getState().screen_buffer.copy()
        except AttributeError:
            screen = self.dummy_screen
        return self.invert_screen(screen)

    def get_obs(self):
        if self.obs_type == 'ram':
            return self.get_ram()
        elif self.obs_type == 'image':
            return self.get_image()

    def isDone(self):
        return self.game.is_episode_finished() or self.game.is_player_dead() or self.getState() is None

    # ===========================================  ==============================================================

    def invert_screen(self, img):
        if self.screen_format in inverted_screen_formats:
            return np.rollaxis(img, 0, 3)
        else:
            return img

    def __delete_conflict_actions(self):
        if self._mode == 'human':
            return
        action_codes_copy = self.available_action_codes.copy()

        print("Initial actions size: " + str(len(action_codes_copy)))
        for i in tqdm.trange(len(self.available_action_codes)):
            action = self.available_action_codes[i]
            ac_names = action_to_buttons(self.allowed_actions, action)

            if all(elem in ac_names for elem in ['MOVE_LEFT', 'MOVE_RIGHT']) or all(
                    elem in ac_names for elem in ['MOVE_BACKWARD', 'MOVE_FORWARD']) or all(
                elem in ac_names for elem in ['TURN_RIGHT', 'TURN_LEFT']) or all(
                elem in ac_names for elem in ['SELECT_NEXT_WEAPON', 'SELECT_PREV_WEAPON']):
                action_codes_copy.remove(action)

        print("Final actions size: " + str(len(action_codes_copy)))
        self.available_action_codes = action_codes_copy

    def __initHumanPlayer(self):
        self._mode = 'human'
        self.__load_level()

        self.game.add_game_args('+freelook 1')
        self.game.set_window_visible(True)
        self.game.set_mode(Mode.SPECTATOR)
        self.is_render_in_human_mode = False

        self.__init_game()

    def advanceAction(self, tick=0):
        try:
            if tick:
                self.game.advance_action(tick)
            else:
                self.game.advance_action()
            return True
        except ViZDoomUnexpectedExitException:
            return False

    def playHuman(self):
        self.__initHumanPlayer()

        while not self.game.is_episode_finished() and not self.game.is_player_dead():
            self.advanceAction()

            state = self.getState()
            if state is None:
                if self.record_file_path is None:
                    self.game.new_episode()
                else:
                    self.game.new_episode(self.record_file_path)
                state = self.getState()

            total_reward = self.game.get_total_reward()
            info = self.get_info()
            info["TOTAL_REWARD"] = round(total_reward, 4)
            print('===============================')
            print('State: #' + str(state.number))
            print('Action: \t' + str(self.game.get_last_action()) + '\t (=> only allowed actions)')
            print('Reward: \t' + str(self.game.get_last_reward()))
            print('Total Reward: \t' + str(total_reward))
            print('Variables: \n' + str(info))
            sleep(0.02857)  # 35 fps = 0.02857 sleep between frames
        print('===============================')
        print('Done')
        return


class DoomFuncWrapper(gym.Wrapper):
    """
    Wrapper of VizDoom basic functions.
    Extend your wrapper with this class to get access to these functions.
    """

    def __init__(self, env):
        """
        :param env: env or env_id
        """
        if isinstance(env, str):
            env = gym.make(env)
        super(DoomFuncWrapper, self).__init__(env)
        self.doom_env = self.unwrapped
        self.screen_resized = None

    def setGame(self, level=None, resolution=None, render=True):
        self.doom_env.set_game(level, resolution, render)

    def advanceAction(self, tick=0):
        return self.doom_env.advanceAction(tick)

    def playHuman(self):
        self.doom_env.playHuman()

    def getState(self):
        return self.doom_env.getState()

    def getGameInfo(self):
        return self.doom_env.get_info()

    def getGameScreen(self):
        return self.doom_env.get_image()

    def getScreenFormat(self):
        return self.doom_env.screen_format

    def getRAM(self):
        return self.doom_env.get_ram()

    def getButtonNames(self, action):
        return self.doom_env.getButtonsNames(action)

    def getEncodedAction(self):
        return self.doom_env.getLastAction()

    def getPossibleActionsCodes(self):
        return self.doom_env.available_action_codes

    def getActionSize(self):
        return len(self.getPossibleActionsCodes())

    def getActionIndexFromEncoding(self, action_code):
        return self.getPossibleActionsCodes().index(action_code)

    def getRandomAction(self):
        return random.choice(self.getPossibleActionsCodes())

    def get_image(self):
        return self.doom_env.get_image()

    def step(self, action):
        return self.doom_env.step(action)

    def reset(self):
        return self.doom_env.reset()

    def render(self, mode='human', **kwargs):
        if 'close' in kwargs and kwargs['close']:
            if self.doom_env.viewer is not None:
                self.doom_env.viewer.close()
                self.doom_env.viewer = None
            return
        if mode == 'human' and not self.doom_env.is_render_in_human_mode:
            return
        img = self.get_image()

        if 'img' in kwargs:
            img = kwargs['img']

        if mode == 'rgb_array':
            return img
        elif mode is 'human':
            if self.doom_env.viewer is None:
                self.doom_env.viewer = rendering.SimpleImageViewer()
            self.doom_env.viewer.imshow(img)
