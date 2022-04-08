from typing import Callable, List, Optional, Tuple, Union
import csv

import gym
import numpy as np

from stable_baselines.common.vec_env import VecEnv


class RewardManipulation:
    def __init__(self, min_r, max_r, clipping, min_max_scaling, scale, buffer_rewards):
        assert (not clipping and not min_max_scaling) or (min_r is not None and max_r is not None)
        self.clipping = clipping
        self.min_max_scaling = min_max_scaling
        self.min_r = min_r
        self.max_r = max_r
        self.scale = scale
        self.buffer = buffer_rewards
        self.neutral_reward = None
        self.r_buffer = None

    def manipulate(self, r, w, done):
        raise NotImplementedError

    def _preprocess(self, r, done):
        if self.buffer:
            r = self._buffer(r, done)
            if not done:
                return r

        if self.min_r is not None:
            # clipping
            if self.clipping:
                r = np.clip(r, self.min_r, self.max_r)
            r = (r - self.min_r) / (self.max_r - self.min_r)
        if self.scale is not None:
            r = r * self.scale
        return r

    def _buffer(self, r, done):
        self.r_buffer = r if (self.r_buffer is None) else self.r_buffer + r
        if done:
            r = self.r_buffer
            self.r_buffer = None

        else:
            r = self.neutral_reward
        return r


class LinearScalarizer(RewardManipulation):
    def __init__(self, min_r=None, max_r=None, clipping=True, min_max_scaling=True, scale=10, buffer_rewards=True):
        super().__init__(min_r, max_r, clipping, min_max_scaling, scale, buffer_rewards=buffer_rewards)
        self.neutral_reward = np.zeros(2)

    def manipulate(self, r, w, done):
        r = self._preprocess(r, done)
        scalarized_r = np.sum(r * w)
        return scalarized_r


class RewardScaler(RewardManipulation):
    def __init__(self, scale=1):
        self.scale = scale

    def manipulate(self, orig_r, w, done):
        r = self.scale * orig_r
        return r


def evaluate_policy(
        model: "BaseRLModel",
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        verbose: bool = True
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths = [], []
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        episode_reward = None
        episode_length = 0
        i = 0
        while not done:
            action, q_vals = model.predict(obs, state=state, deterministic=deterministic)
            obs_ = obs
            obs, reward, done, _info = env.step(action)
            if q_vals is not None and type(q_vals) is dict:
                all_rs = False
                if all_rs:
                    print(f"{q_vals['thr']} Q{i} (r,l,d,u): {q_vals['q']} | qt {q_vals['qt']} | qc {q_vals['qc']}")
                else:
                    if verbose and episode_reward is None:
                        # if verbose:
                        print(f"-----------------------------------------")
                        print(
                            f"{q_vals['thr']} first q-vals (r,l,d,u): {q_vals['q']} | qt {q_vals['qt']} | qc {q_vals['qc']}")
                    if verbose and done:
                        print(
                            f"{q_vals['thr']} last q-vals (r,l,d,u): {q_vals['q']} | qt {q_vals['qt']} | qc {q_vals['qc']} r:{reward}")
            i += 1
            reward = np.array(reward)
            if episode_reward is None:
                episode_reward = reward
            else:
                episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: {:.2f} < {:.2f}".format(mean_reward,
                                                                                                     reward_threshold)
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


class CSVLogger(object):
    def __init__(self, outfile):
        self.file = outfile
        self._log_dict = {}
        self._log_archive = []

    def set_value(self, key, value):
        self._log_dict[key] = value

    def set_values(self, value_dict):
        self._log_dict.update(value_dict)

    def write_log(self):
        # get log-file keys
        csv_keys = set()
        if self.file.exists():
            with open(self.file, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                try:
                    csv_keys = set(next(reader))
                except Exception:
                    pass
        # rewrite csv if unseen keys
        if csv_keys < set(self._log_dict.keys()):
            with open(self.file, 'w+', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, self._log_dict.keys())
                writer.writeheader()
                writer.writerows(self._log_archive)
                writer.writerow(self._log_dict)
        # append csv if no unseen keys
        else:
            with open(self.file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, self._log_dict.keys())
                writer.writerow(self._log_dict)

        self._log_archive.append(self._log_dict)
        self._log_dict = {k: None for k in self._log_dict.keys()}
