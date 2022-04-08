from gym.envs.registration import register
from gtlo.fruityGym import FruityGymEnv

register(
    id=FruityGymEnv.ENV_ID,
    entry_point='gtlo.fruityGym:FruityGymEnv',
    max_episode_steps=2500,
)
