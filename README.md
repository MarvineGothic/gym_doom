# gym_doom
Gym environment for ViZDOOM

### Requirements
 - [ViZDoom](https://github.com/mwydmuch/ViZDoom)
 - Python 3.6.8 (ViZDoom dependency)

### Configuration
#### 1. Environments 
You can create two types of environments:

`Doom-ram-v0` - It will have pseudo "RAM" observations. As "RAM" it uses `available_game_variables` from level config file.

`Doom-v0` - Environment with screen images as observations.

##### Default environtment:
```
env = gym.make(env_id)
```

This will create default game with `deathmatch` level and `640X480` resolution
#### 2. Wrappers
##### DoomFuncWrapper
`:param env: env or env_id`

Create environment:
```from gym_doom.doom_env import DoomFuncWrapper
env = gym.make('Doom-v0')
env = DoomFuncWrapper(env)
```
or
```
env = DoomFuncWrapper('Doom-v0')
```

Extends environment to getting access to basic functions:

    def setGame(self, level=None, resolution=None, render=True):

    def advanceAction(self, tick=0):

    def playHuman(self):

    def getState(self):

    def getGameInfo(self):

    def getGameScreen(self):

    def getScreenFormat(self):

    def getRAM(self):

    def getButtonNames(self, action):

    def getEncodedAction(self):

    def getPossibleActionsCodes(self):

    def getActionSize(self):

    def getActionIndexFromEncoding(self, action_code):

    def getRandomAction(self):

    def get_image(self):

##### ViZDoomEnv
```
:param env: or env_id
:param level: name of level without extension
:param data_dir: path to scenarios directory
```
Inherited from DoomFuncWrapper. Main wrapper to change game to your custom level. Path directory must include `/scenarios/`
with `level_name.cfg` and `level_name.wad` files. Therefore you must provide same name for level
parameter as in `level_name` without extension.


Create environment:
```
dir_ = "E:\VizDOOM"
env = gym.make('Doom-v0')
env = ViZDoomEnv(env, level='deadly_corridor', data_dir=dir_)
```

##### ScreenWrapper
```
:param env: VizDoom environment or env_id
:param dim: resize dimensions as a tuple
:param resolution: desired resolution
:param render: renders the main window
:param dummy_obs: fills observations with zeros
```
Inherited from ObservationWrapper and DoomFuncWrapper. 

To resize observation space and pygame window provide `dim` in form of tuple with new dimensions. 
F.ex. `dim=(100, 100)`

To change resolution of main VizDoom screen specify `resolution="160x120"`

To disable rendering of main VizDoom window set `render=False`
 
 List of available resolutions:
 
        '160x120', '200x125', '200x150', '256x144', '256x160', '256x192', '320x180', '320x200',
               '320x240', '320x256', '400x225', '400x250', '400x300', '512x288', '512x320', '512x384',
               '640x360', '640x400', '640x480', '800x450', '800x500', '800x600', '1024x576', '1024x640',
               '1024x768', '1280x720', '1280x800', '1280x960', '1280x1024', '1400x787', '1400x875',
               '1400x1050', '1600x900', '1600x1000', '1600x1200', '1920x1080'

```
env = gym.make('Doom-v0')
env = ViZDoomEnv(env, level='deadly_corridor', data_dir=dir_)
env = ScreenWrapper(env, dim=(100, 100),  render=False, dummy_obs=True)
```

#### 3. Python version issue
ViZDoom can probably also run successfully on other Python3 versions. 
Try to edit `_COMPILED_PYTHON_VERSION` in your `vizdoom/__init__.py` to your version:

```
import sys as _sys

_COMPILED_PYTHON_VERSION = "3.6.6"

_this_python_version = "{}.{}.{}".format(*_sys.version_info[0:3])

if _COMPILED_PYTHON_VERSION != _this_python_version:
    raise SystemError(
        "This interpreter version: '{}' doesn't match with version of the interpreter ViZDoom was compiled with: {}".format(
            _this_python_version, _COMPILED_PYTHON_VERSION))

from .vizdoom import __version__ as __version__
from .vizdoom import *

import os as _os

scenarios_path = _os.path.join(__path__[0], "scenarios")
wads = [wad for wad in sorted(_os.listdir(scenarios_path)) if wad.endswith(".wad")]
configs = [cfg for cfg in sorted(_os.listdir(scenarios_path)) if cfg.endswith(".cfg")]
```


### Running 
ViZDoomGymEnv method will create a gym environment.

##### Human player

```
import gym
from gym_doom.wrappers import ViZDoomEnv, ScreenWrapper

if __name__ == '__main__':

    viz_doom_renderer = False
    dir_ = "E:\VizDOOM"
    env = gym.make('Doom-ram-v0')
    env = ViZDoomEnv(env, level='deadly_corridor', data_dir=dir_)
    env = ScreenWrapper(env, dim=(100, 100),  render=True, dummy_obs=True)

    env.playHuman()

```
##### Loop for RL
```
import random
import time
import gym
from gym_doom.wrappers import ViZDoomEnv, ScreenWrapper


def randomAction(env):
    actions = env.unwrapped.available_action_codes
    return random.choice(actions)


if __name__ == '__main__':

    viz_doom_renderer = False
    dir_ = "E:\VizDOOM"
    env = gym.make('Doom-ram-v0')
    env = ViZDoomEnv(env, level='deadly_corridor', data_dir=dir_)
    env = ScreenWrapper(env, dim=(100, 100),  render=True, dummy_obs=True)

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
```

 
### Links
[ViZDoom](https://github.com/mwydmuch/ViZDoom) - original ViZDoom with all Documentation and Installation guide

[ViZDoom Tutorial](http://vizdoom.cs.put.edu.pl/tutorial) - includes essential information about installation, configuration and running ViZDoom

[ppaquette_gym_doom](https://github.com/ppaquette/gym-doom) - outdated original repository for gym ViZDoom environment

