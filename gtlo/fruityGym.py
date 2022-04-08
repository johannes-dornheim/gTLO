import numpy as np
from gym import Env
from gym.core import Wrapper
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import fruit.envs.games.deep_sea_treasure.engine as fruitengine
from fruit.envs.games.deep_sea_treasure.engine import DeepSeaTreasure, DeepSeaTreasureConstants
from fruit.envs.games.deep_sea_treasure.sprites import TreasureSprite
import pygame
import os


class FruityGymEnv(Env):
    ENV_ID = 'fruity-gym-v0'

    def __init__(self, fruit_env):
        self.fruit_env = fruit_env
        self.graphical_state = fruit_env.game.graphical_state
        # Set these in ALL subclasses
        self.action_space = self._to_gym_space(fruit_env.get_action_space())
        self.observation_space = self._to_gym_space(fruit_env.get_state_space())

        # Set this in SOME subclasses
        # metadata = {'render.modes': []}
        # reward_range = (-float('inf'), float('inf'))
        # spec = None

    def step(self, action):
        r = self.fruit_env.step(action)
        s = self.fruit_env.get_state()
        if self.graphical_state:
            # convert fruitAPI image to stable-baselines style
            s = s.reshape(list(s.shape) + [1])
        done = self.fruit_env.is_terminal()
        return s, r, done, {}

    def reset(self):
        self.fruit_env.reset()
        s = self.fruit_env.get_state()
        if self.graphical_state:
            # convert fruitAPI image to stable-baselines style
            s = s.reshape(list(s.shape) + [1])
        return s

    def render(self, mode='human'):
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        return

    def _is_done(self):
        return self.fruit_env.is_terminal()

    def _to_gym_space(self, fruit_space):
        if type(fruit_space) == list:
            assert False, "Make sure to apply a FruitAPI-Processor for graphical state"
        elif fruit_space.is_discrete() and len(fruit_space.get_shape()) == 1:
            return Discrete(fruit_space.get_max() + 1)
        elif self.graphical_state:
            fs_shape = fruit_space.get_shape()
            return Box(0, 255, shape=(fs_shape[0], fs_shape[1], 1))
        else:
            raise NotImplementedError


class NamedDeepSeaTreasure(DeepSeaTreasure):
    """
    Pre-configured named DST Environments used in different related papers
    """

    def __init__(self, width, named_game, min_depth=3, min_vertical_step=1, max_vertical_step=3, transition_noise=0.,
                 reward_noise=0., front_shape=DeepSeaTreasureConstants.CONCAVE, seed=None, min_treasure=1,
                 max_treasure=1000, render=False, speed=60, agent_random_location=False,
                 reshape_reward_weights=None, is_debug=False, graphical_state=False):
        self.num_of_cols = width
        self.min_depth = min_depth
        self.min_vertical_step = min_vertical_step
        self.max_vertical_step = max_vertical_step
        self.min_treasure = min_treasure
        self.max_treasure = max_treasure
        self.depths = [min_depth] * self.num_of_cols
        self.steps = [min_depth] * self.num_of_cols
        self.agent_row = 0
        self.agent_col = 0
        self.total_steps = 0
        self.weights = reshape_reward_weights
        self.chose_weight = 0
        step_range = max_vertical_step - min_vertical_step + 1
        self.trs = [None for _ in range(width)]
        self.is_debug = is_debug
        self.log_freq = 200
        self.graphical_state = graphical_state
        self.total_score = 0
        self.total_score_2 = 0
        self.extra_col_depths = []
        if seed is None or seed < 0 or seed >= 9999:
            if seed is not None and (seed < 0 or seed >= 9999):
                print("Invalid seed ! Default seed = randint(0, 9999")
            self.seed = np.random.randint(0, 9999)
            self.random_seed = True
        else:
            self.random_seed = False
            self.seed = seed
            np.random.seed(seed)

        self.agent_random = agent_random_location
        if self.agent_random:
            self.agent_col = np.random.randint(0, self.num_of_cols - 1)
        self.named_game = named_game
        if named_game is 'van2014multi':
            """
            @article{van2014multi,
                title={Multi-objective reinforcement learning using sets of pareto dominating policies},
                author={Van Moffaert, Kristof and Now{\'e}, Ann},
                journal={The Journal of Machine Learning Research},
                year={2014}
            }
            """
            assert self.max_treasure == 124
            assert self.min_depth == 1
            assert min_vertical_step == 0
            assert width == 10

            self.treasures = [1, 2, 3, 5, 8, 16, 24, 50, 74, 124]
            self.depths = [1, 2, 3, 4, 4, 4, 7, 7, 9, 10]
            self.num_of_rows = self.depths[-1] + 1
            for i in range(len(self.depths)):
                self.steps[i] = -(self.depths[i] + i)

            self.front_shape = DeepSeaTreasureConstants.CONCAVE

        self.transition_noise = transition_noise
        self.reward_noise = reward_noise
        self.num_of_actions = 4
        self.transition_function = self._DeepSeaTreasure__set_transitions()
        self.pareto_solutions = self._DeepSeaTreasure__get_pareto_solutions()
        self.num_of_objectives = 2
        self.rd = render
        if self.num_of_cols > 60:
            if self.rd:
                print("Could not render when width > 60")
                self.rd = False
        self.screen = None
        self.speed = speed
        self.current_path = os.path.dirname(os.path.abspath(fruitengine.__file__))
        self.width = 50 * self.num_of_cols
        self.height = int((self.width / self.num_of_cols) * self.num_of_rows)
        self.tile_width = int(self.width / self.num_of_cols)
        self.player_width = int(self.tile_width * 5 / 6)
        self.player_height = int(self.player_width * 55 / 76)
        self.sprites = pygame.sprite.Group()
        self.current_buffer = np.array([[[0, 0, 0] for _ in range(self.height)] for _ in range(self.width)])

        # Initialize
        self._DeepSeaTreasure__init_pygame_engine()
        self._DeepSeaTreasure__generate_submarine()
        self._DeepSeaTreasure__generate_treasures()

        # Render the first frame
        self._DeepSeaTreasure__render()

    """
    (ugly) overwrites due to adoptions needed to create the extra column in reymond2021  
    """

    def is_terminal(self):
        # print(self.total_steps, self.max_steps)
        # if self.total_steps > self.max_steps:
        #    return True
        is_extra_col = self.agent_col >= len(self.depths)

        if not is_extra_col and self.agent_row == self.depths[self.agent_col]:
            return True
        else:
            return False

    def _DeepSeaTreasure__generate_treasures(self):
        color_blue = (100, 100, 100)
        color_black = (0, 0, 0)
        offset = int((self.tile_width - self.player_width) / 2)
        image = pygame.image.load(self.current_path + "/graphics/treasure.png")
        if self.rd:
            image = pygame.transform.scale(image, (self.player_width, self.player_width)).convert_alpha()
        else:
            image = pygame.transform.scale(image, (self.player_width, self.player_width))

        for row in range(self.num_of_rows):
            for col in range(self.num_of_cols):
                is_extra_col = col >= len(self.depths)
                if is_extra_col or row <= self.depths[col]:
                    pygame.draw.rect(self.screen, color_blue,
                                     pygame.Rect(col * self.tile_width, row * self.tile_width,
                                                 self.tile_width - 1, self.tile_width - 1))
                    if not is_extra_col and row == self.depths[col]:
                        self.trs[col] = TreasureSprite(pos_x=col * self.tile_width + offset,
                                                       pos_y=row * self.tile_width + offset, sprite_bg=image)
                        self.sprites.add(self.trs[col])
                else:
                    pygame.draw.rect(self.screen, color_black,
                                     pygame.Rect(col * self.tile_width, row * self.tile_width,
                                                 self.tile_width, self.tile_width))

    def _DeepSeaTreasure__get_transition(self, col, row, action):
        new_row = row
        new_col = col
        depths = self.depths + self.extra_col_depths

        if row > depths[col]:
            return col * self.num_of_rows + row

        # Right
        if action == 0:
            new_col = new_col + 1
            if new_col > self.num_of_cols - 1:
                new_col = new_col - 1

        # Left
        if action == 1:
            new_col = new_col - 1
            if new_col < 0:
                new_col = 0
            elif not self._DeepSeaTreasure__is_valid(new_row, new_col):
                new_col = new_col + 1

        # Down
        if action == 2:
            new_row = new_row + 1
            if new_row > depths[col]:
                new_row = new_row - 1

        # Up
        if action == 3:
            new_row = new_row - 1
            if new_row < 0:
                new_row = 0

        return new_col * self.num_of_rows + new_row

    def _DeepSeaTreasure__get_pareto_solutions(self):
        solutions = []
        if self.transition_noise == 0.0:
            for i in range(self.num_of_cols - len(self.extra_col_depths)):
                solutions.append([self.treasures[i], self.steps[i]])
            return solutions
        else:
            return None

    def _DeepSeaTreasure__check_reward(self):
        col = self.agent_col
        row = self.agent_row
        objs = [0., -1.]
        is_extra_col = col >= len(self.depths)

        if not is_extra_col and row == self.depths[col]:
            objs[0] = self.treasures[col]
        self.total_score = int(self.total_score + objs[0])
        self.total_score_2 = int(self.total_score_2 + objs[1])
        if self.weights is not None:
            w = self.weights[self.chose_weight]
            return np.multiply(objs, w)
        else:
            return objs


class PositionStateWrapper(Wrapper):
    """
    state-tuple as position instead of image
    """

    def __init__(self, env):
        print('one hot vector representation seems to work much better in tabular case!')
        super().__init__(env)
        assert not env.unwrapped.graphical_state, 'Position-State wrapper is meant for discrete state env'
        fruit_env = env.unwrapped.fruit_env
        new_obs_space = Box(np.array([0, 0]),
                            np.array([fruit_env.get_processor().num_of_rows, fruit_env.get_processor().num_of_cols]),
                            shape=(2,))
        self.observation_space = new_obs_space
        env_ = self
        while hasattr(env_, 'env'):
            env_.env.observation_space = new_obs_space
            env_ = env_.env

    def step(self, action):
        s, r, done, info = self.env.step(action)
        s = self._create_position_state()
        return s, r, done, info

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._create_position_state()

    def _create_position_state(self):
        fruit_env = self.env.unwrapped.fruit_env
        return np.array([fruit_env.get_processor().agent_row, fruit_env.get_processor().agent_col])


class StaticScalarizationWrapper(Wrapper):
    """
    Wraps the MORL environment with linear-scalarization, such that it can be used as an single-objective env.
    """

    def __init__(self, env, configuration):
        self.configuration = configuration
        super().__init__(env)

    def step(self, action):
        s, r, done, info = self.env.step(action)
        r = np.mean(np.array(self.configuration) * r)
        return s, r, done, info


class DynamicConfigurationWrapper(Wrapper):  # todo refactor this thing!
    def __init__(self, env, initial_configuration, configurations=None, reward_manipulator=None,
                 manipulate_state=True, step_state=False, max_time_steps=50):
        self.configuration = initial_configuration
        self.next_configuration = self.configuration
        super().__init__(env)
        self.manipulate_state = manipulate_state
        self.reward_manipulator = reward_manipulator
        self.step_state = step_state
        self.steps = 0

        if manipulate_state:
            # overwrite observation-space, such that the configuration is part of the observation
            unwr = self.env.unwrapped
            if unwr.ENV_ID in ['2d-deepdrawing-5ts-stressstate-v0', '2d-deepdrawing-5ts-stressoffsetstate-v0'] \
                    or unwr.graphical_state:
                # add extra information to an extra row in image-tensor
                new_shape = (unwr.observation_space.shape[0] + 1,
                             unwr.observation_space.shape[1],
                             unwr.observation_space.shape[2])
                new_obs_space = Box(low=0, high=255, shape=new_shape)
            else:
                #  set observation space self.env.unwrapped.
                new_shape = (unwr.observation_space.shape[0] + len(self.configuration),)
                min_cfg, max_cfg = np.array(configurations).min(), np.array(configurations).max()

                new_low = np.concatenate([unwr.observation_space.low, np.array([min_cfg,
                                                                                min_cfg])])
                new_high = np.concatenate([unwr.observation_space.high, np.array([max_cfg,
                                                                                  max_cfg])])
                if self.step_state:
                    new_shape = (new_shape[0] + 1,)
                    new_low = np.append(new_low, 0)
                    new_high = np.append(new_high, max_time_steps)

                new_obs_space = Box(new_low, new_high, shape=new_shape)
            self.observation_space = new_obs_space
            env_ = self
            while hasattr(env_, 'env'):
                # todo not a good solution, can cause problems when observation space is used internally by the env...
                env_.env.observation_space = new_obs_space
                env_ = env_.env

    def step(self, action):
        s, r_, done, info = self.env.step(action)
        r = np.array(r_)

        w = np.array(self.configuration)
        if self.reward_manipulator is not None:
            r = self.reward_manipulator.manipulate(r, w, done)

        self.steps += 1
        if done:
            self.steps = 0
        if self.manipulate_state:
            s = self._manipulate_state(s)
        return s, r, done, info

    def reset(self, **kwargs):
        self.configuration = self.next_configuration
        s = self.env.reset(**kwargs)
        if self.manipulate_state:
            s = self._manipulate_state(s)
        return s

    def set_next_configuration(self, configuration):
        self.next_configuration = configuration

    def _manipulate_state(self, s):
        unwr = self.env.unwrapped
        if unwr.ENV_ID in ['2d-deepdrawing-5ts-stressstate-v0', '2d-deepdrawing-5ts-stressoffsetstate-v0'] \
                or unwr.graphical_state:
            # add configuration to an extra row of the image-tensor

            config_column = np.zeros((s.shape[1], s.shape[2]))
            i = 0
            for c in self.configuration:
                config_column[i, 0] = c
                i += 1
            """
            if self.step_state:
                config_column[i] = self.steps
            """
            # print(f's-shape {s.shape}, config shape {config_column.shape}, config {self.configuration}, obs-space shape {unwr.observation_space.shape}')
            s = np.append(s, config_column).reshape(unwr.observation_space.shape)
            return s
        else:
            s = np.concatenate((s, self.configuration))
            if self.step_state:
                s = np.append(s, self.steps)
            return s
