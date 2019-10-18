from gym.envs.registration import register
from .doom_env import DoomEnv
# Env registration
# ==========================

# for level in ['basic', 'deadly_corridor', 'deathmatch', 'defend_the_center', 'defend_the_line',
#               'health_gathering', 'my_way_home', 'predict_position', 'take_cover']:
for obs_type in ['image', 'ram']:
    name = 'Doom'
    if obs_type == 'ram':
        name = '{}-ram'.format(name)

    register(
        id='{}-v0'.format(name),
        entry_point='gym_doom:DoomEnv',
        kwargs={'obs_type': obs_type},
        max_episode_steps=10000,
        reward_threshold=10.0,
    )

    register(
        id='{}-v4'.format(name),
        entry_point='gym_doom:DoomEnv',
        kwargs={'obs_type': obs_type},
        max_episode_steps=100000,
        reward_threshold=10.0,
    )

    # register(
    #     id='Doom-v0',
    #     entry_point='gym_doom:DoomEnv',
    #     max_episode_steps=10000,
    #     reward_threshold=10.0,
    # )
