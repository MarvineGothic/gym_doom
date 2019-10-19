import os
import sys

import vizdoom
from termcolor import colored

from .utils import *


class Config:
    is_initialized = False
    PROJECT_DIR = ""
    RECORD_FILE = "/records/record"
    RECORD_FILE_PATH = ""
    VIZDOOM_SCENARIO_PATH = ""
    VIZDOOM_SCENARIO_FOLDER = "/scenarios/"

    VIZDOOM_DIR_PATH = os.path.dirname(vizdoom.__file__)
    VIZDOOM_PATH = VIZDOOM_DIR_PATH + '/vizdoom.exe'
    FREEDOOM_PATH = VIZDOOM_DIR_PATH + '/freedoom2.wad'

    DEFAULT_SCREEN_RESOLUTION = "640X480"

    ACTIONS_SET = [
        "ATTACK",
        "USE",
        "JUMP",
        "CROUCH",
        "TURN180",
        "ALTATTACK",
        "RELOAD",
        "ZOOM",
        "SPEED",
        "STRAFE",
        "MOVE_RIGHT",
        "MOVE_LEFT",
        "MOVE_BACKWARD",
        "MOVE_FORWARD",
        "TURN_RIGHT",
        "TURN_LEFT",
        "LOOK_UP",
        "LOOK_DOWN",
        "MOVE_UP",
        "MOVE_DOWN",
        "LAND",
        "SELECT_WEAPON1",
        "SELECT_WEAPON2",
        "SELECT_WEAPON3",
        "SELECT_WEAPON4",
        "SELECT_WEAPON5",
        "SELECT_WEAPON6",
        "SELECT_WEAPON7",
        "SELECT_WEAPON8",
        "SELECT_WEAPON9",
        "SELECT_WEAPON0",
        "SELECT_NEXT_WEAPON",
        "SELECT_PREV_WEAPON",
        "DROP_SELECTED_WEAPON",
        "ACTIVATE_SELECTED_WEAPON",
        "SELECT_NEXT_ITEM",
        "SELECT_PREV_ITEM",
        "DROP_SELECTED_ITEM",
        "LOOK_UP_DOWN_DELTA",
        "TURN_LEFT_RIGHT_DELTA",
        "MOVE_FORWARD_BACKWARD_DELTA",
        "MOVE_LEFT_RIGHT_DELTA",
        "MOVE_UP_DOWN_DELTA"
    ]

    CONFIG = 0
    SCENARIO = 1
    MAP = 2
    DIFFICULTY = 3
    ACTIONS = 4
    GAME_VARIABLES = 5

    DOOM_SETTINGS = {}

    @staticmethod
    def init(level, project_dir=None):
        level = level.split('.')[0]
        if Config.is_initialized and project_dir is None:
            return
        if Config.is_initialized and project_dir is not None:
            Config.is_initialized = False

        if Config.is_initialized:
            return
        if project_dir is None:
            Config.PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
            Config.RECORD_FILE_PATH = None
        else:
            Config.PROJECT_DIR = project_dir
            Config.RECORD_FILE_PATH = Config.PROJECT_DIR + Config.RECORD_FILE

        Config.VIZDOOM_SCENARIO_PATH = Config.PROJECT_DIR + Config.VIZDOOM_SCENARIO_FOLDER

        cfg_files = []
        wad_files = []
        for file_name in os.listdir(Config.VIZDOOM_SCENARIO_PATH):
            f_name, ext = file_name.split('.')
            if ext == 'cfg':
                cfg_files.append(file_name)
            if ext == 'wad':
                wad_files.append(f_name)

        if len(wad_files) == 0:
            print(colored("WARNING! Level {} doesn't exist. \n\tAvailable levels are: \n\t\t{}".
                          format(level, '\n\t\t'.join(wad_files)), 'red'))

            sys.exit()

        for file_name in cfg_files:
            f_name, ext = file_name.split('.')
            if ext == 'cfg':
                if f_name not in wad_files:
                    print(colored("WARNING! {} should contain .wad file with same name. \n\t\t File {} is not loaded.".
                                  format(file_name, file_name), 'red'))
                else:
                    try:
                        parsed_file = CFG_Parser.parse(os.path.join(Config.VIZDOOM_SCENARIO_PATH, file_name))
                        Config.DOOM_SETTINGS[f_name] = [f_name + '.cfg', f_name + '.wad', 'map01', 5,
                                                        parsed_file['available_buttons'],
                                                        parsed_file['available_game_variables']]
                    except Exception as e:
                        print(colored("WARNING! File {} could not be loaded. \n\t\tMissing {} in configuration.".
                                      format(file_name, e), 'red'))
                        pass

        Config.is_initialized = True
