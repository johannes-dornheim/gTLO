from time import sleep

import gym
from fruit.envs.games.deep_sea_treasure.engine import DeepSeaTreasure
from fruit.envs.juice import FruitEnvironment
from fruit.state.processor import AtariProcessor
from gym import logger

from gtlo.fruityGym import PositionStateWrapper


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='ms-evolution-rd-v0', help='Select the environment to run')
    args = parser.parse_args()
    """

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    game = DeepSeaTreasure(graphical_state=True, width=10, seed=100, render=True, max_treasure=124, speed=1000,
                           min_depth=1, min_vertical_step=0, named_game='van2014multi')

    fruit_env = FruitEnvironment(game, max_episode_steps=60, state_processor=AtariProcessor())

    env = gym.make('fruity-gym-v0', fruit_env=fruit_env)
    env = PositionStateWrapper(env)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    agent = RandomAgent(env.action_space)

    episode_count = 10000
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            env.render()
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            print(f'a {action}, ob {ob}, r {reward}, d {done}')
            if done:
                break
            sleep(2)

    # Close the env and write monitor result info to disk
    env.close()
