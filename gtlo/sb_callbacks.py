import numpy as np
from stable_baselines.common.callbacks import BaseCallback
from gtlo.helpers import CSVLogger, evaluate_policy
import warnings

from fruit.utils.hypervolume import HVCalculator


class MORLEvaluation(BaseCallback):
    def __init__(self,
                 eval_env,
                 configurations,
                 manipulator,
                 hypervolume_ref_point=None,
                 n_eval_episodes=1,
                 eval_freq=100,
                 logdir=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.eval_env = eval_env
        self.configurations = configurations
        self.eval_freq = eval_freq
        self.curr_episode = 0
        self.n_eval_episodes = n_eval_episodes
        self.manipulator = manipulator
        if hypervolume_ref_point is None:
            self.hypervolume_ref_point = [0, 0]
        else:
            self.hypervolume_ref_point = hypervolume_ref_point
        # calc hypervolume vol for pareto-front

        try:
            pareto_front = eval_env.unwrapped.fruit_env.game.get_pareto_solutions()
            self.pareto_hv = HVCalculator.get_volume_from_array(pareto_front, self.hypervolume_ref_point)
        except:
            self.pareto_hv = None

        self.csv_logger = None
        if logdir:
            self.csv_logger = CSVLogger(logdir.joinpath('MORL_log.csv'))
            self.detailed_logger = CSVLogger(logdir.joinpath('MORL_detail_log.csv'))

    def _on_step(self) -> bool:
        if self._check_if_eval_episode():
            # EVALUATION
            front = []
            for i, configuration in enumerate(self.configurations):
                # print("========================================")
                # print(f"eval configuration {configuration}")
                self.eval_env.set_next_configuration(configuration)
                self.eval_env.reset()
                episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
                                                                   n_eval_episodes=self.n_eval_episodes,
                                                                   render=False,
                                                                   deterministic=True,
                                                                   return_episode_rewards=True)
                if len(episode_rewards) == 1:
                    scaled_r = self.manipulator.manipulate(episode_rewards[0], configuration, done=True)
                    episode_rewards = episode_rewards[0]
                    front.append(episode_rewards)
                else:
                    scaled_r = [self.manipulator.manipulate(r, configuration, done=True) for r in episode_rewards]
                    episode_rewards = np.array(episode_rewards)
                    mean_e_reward = episode_rewards.mean(axis=0)
                    front.append(mean_e_reward)
                # print(f'\nconf:\t{configuration}\tr:{episode_rewards}\t')
                # print(f'sr:{scaled_r}')
                if self.detailed_logger is not None:
                    self.detailed_logger.set_values({'episode': self.curr_episode,
                                                     'config': configuration,
                                                     'r': episode_rewards,
                                                     'scaled_r': scaled_r})
                    self.detailed_logger.write_log()

            # hypervolume vol
            if len(episode_rewards) == 1:
                self.hv = HVCalculator.get_volume_from_array(front, self.hypervolume_ref_point)
                if self.csv_logger is not None:
                    self.csv_logger.set_values({'episode': self.curr_episode, 'step': self.num_timesteps})
                    self.csv_logger.set_values({'hvt': self.hv, 'rel_hvt': self.hv / self.pareto_hv})
                    self.csv_logger.write_log()
                print(f'hvt:{self.hv}, pareto-hv:\t{self.pareto_hv}\nhv/pareto-hv:\t{self.hv / self.pareto_hv}')
            else:
                self.hv = HVCalculator.get_volume_from_array(front, self.hypervolume_ref_point)
                if self.csv_logger is not None:
                    self.csv_logger.set_values({'episode': self.curr_episode, 'step': self.num_timesteps})
                    self.csv_logger.set_values({'hvt': self.hv})
                    self.csv_logger.write_log()
                print(f'hvt:{self.hv}')

        return True

    def _check_if_eval_episode(self):
        self.curr_episode = len(self.locals['episode_rewards']) - 1

        eval = self.n_calls % self.eval_freq == 0 and self.n_calls > 0

        return eval


class MORLEvaluationStochastic(MORLEvaluation):

    def __init__(self, eval_frictions, logdir, **kwargs):
        super().__init__(logdir=logdir, **kwargs)
        self.eval_frictions = eval_frictions

        self.multi_csv_logger = None
        if logdir:
            self.multi_csv_logger = CSVLogger(logdir.joinpath('multiMORL_log.csv'))

    def _on_step(self) -> bool:
        if self._check_if_eval_episode():
            # for frictions
            hvs = []
            for fric in self.eval_frictions:
                # set fric
                self.eval_env.bypass_fric(fric)
                # run eval
                super()._on_step()
                # store hv
                hvs.append(self.hv)
            # calc multi-hv
            multiHV = sum(hvs) / len(hvs)
            print(f"MultiHV {multiHV} | {hvs}")
            self.multi_csv_logger.set_values({'multiHV': multiHV, 'hvs': hvs})
            self.multi_csv_logger.write_log()
        return True


class RandomConfigurationSetter(BaseCallback):
    def __init__(self, configurations, scal_wrapper, **kwargs):
        super().__init__(**kwargs)
        self.configurations = configurations
        self.scal_wrapper = scal_wrapper
        self.curr_configuration = None
        self._set_random_configuration()

    def _on_step(self) -> bool:
        training_env = self.locals['self'].env.unwrapped
        if training_env._is_done():
            self._set_random_configuration()
        return True

    def _set_random_configuration(self):
        w = self.configurations[np.random.choice(len(self.configurations)), :]
        self.scal_wrapper.set_next_configuration(w)
        self.curr_configuration = w
