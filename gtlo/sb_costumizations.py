import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from functools import partial
import gym

from stable_baselines import logger
from stable_baselines.common import tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.buffers import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc
from stable_baselines.deepq.policies import register_policy
from stable_baselines.common.policies import nature_cnn
from stable_baselines import DQN
from stable_baselines.deepq.policies import DQNPolicy

from gtlo.build_graph_conditioned_extension import build_train


def gtlo_dst(scaled_images, **kwargs):
    activ = tf.nn.relu
    extra_features, scaled_images = scaled_images[:, -1], scaled_images[:, :-1]
    extra_features = extra_features[:, :2]
    extra_features = tf.reshape(extra_features, [-1, 2])

    # feature extraction module
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_4 = conv_to_fc(layer_3)
    layer_5 = activ(linear(layer_4, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))
    # head a: conditioned objective (steps)
    layer_6a = activ(
        linear(tf.concat(values=[layer_5, extra_features], axis=1), 'fc2c', n_hidden=128, init_scale=np.sqrt(2)))
    layer_7a = activ(linear(layer_6a, 'fc3c',n_hidden=64, init_scale=np.sqrt(2)))
    # head b: unconditioned objective (treasures)
    layer_6b = activ(linear(layer_5, 'fc2uc',n_hidden=128, init_scale=np.sqrt(2)))

    return layer_7a, layer_6b, layer_6a, extra_features


def gtlo_deepdrawing2rts(scaled_images, **kwargs):
    activ = tf.nn.relu
    extra_features, scaled_images = scaled_images[:, -1, 0, :2], scaled_images[:, :-1]

    # feature extraction module
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2),
                         **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_2)
    layer_4 = activ(linear(layer_3, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))
    # head a: conditioned objective (rt_feeding)
    layer_5a = activ(
        linear(tf.concat(values=[layer_4, extra_features], axis=1),
               'fc2c', n_hidden=128, init_scale=np.sqrt(2)))
    layer_6a = activ(linear(layer_5a, 'fc3c', n_hidden=64, init_scale=np.sqrt(2)))
    # head b: unconditioned objective (rt_thickness)
    layer_5b = activ(linear(layer_4, 'fc2uc', n_hidden=128, init_scale=np.sqrt(2)))
    return layer_6a, layer_5b, layer_5a, extra_features


class GtloPolicy(DQNPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn",
                 obs_phs=None, layer_norm=False, dueling=True, act_fun=tf.nn.relu, **kwargs):
        super(GtloPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                n_batch, dueling=dueling, reuse=reuse,
                                                scale=(feature_extraction == "cnn"), obs_phs=obs_phs)

        self._kwargs_check(feature_extraction, kwargs)

        if layers is None:
            layers = [64]

        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("action_value"):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                    qc_out, qt_out, q_united_out, extra_features = extracted_features
                else:
                    raise NotImplementedError
            if self.generalizedTLO:
                # todo curr. implemented for 2 objectives only
                self.qc = tf_layers.fully_connected(qc_out, num_outputs=self.n_actions, activation_fn=None)
                self.qt = tf_layers.fully_connected(qt_out, num_outputs=self.n_actions, activation_fn=None)

                # ============ extract threshold from self.processed_obs ============
                # very specific to envs used
                """
                graphical = (self.processed_obs[0].shape[0] in [85, 6])
                if graphical:
                    thresholds = self.processed_obs[:, -1, 0] * 256 # revert scaling!
                else:
                    thresholds = self.processed_obs[:, 2]
                self.thresh = thresholds
                """
                thresholds = tf.reshape(extra_features[:, 0], [-1, 1]) * 255
                self.thresh = thresholds

                # ============ TLO action selection by a manipulated q-function ============
                # if max qt < threshold
                # self.q_values = qt
                # else:
                # for a in A:
                # self.q_values(a) = qc if qt(a) >= threshold, else min(qc)

                # ============ max qt (equivalent to state-values) ============
                state_values = tf.multiply(tf.ones_like(self.qt),
                                           tf.reshape(tf.math.reduce_max(self.qt, reduction_indices=[1]), [-1, 1]))
                self.state_vals = state_values

                # ============ build masks for cases and loops ============
                if_mask = tf.where(state_values <= thresholds, tf.ones_like(self.qt), tf.zeros_like(self.qt))
                else_mask = tf.where(state_values > thresholds, tf.ones_like(self.qt), tf.zeros_like(self.qt))
                for_mask = tf.where(self.qt > thresholds, tf.ones_like(self.qt), tf.zeros_like(self.qt))

                # ============ build mask for qt-cases ============
                qt_mask = if_mask

                # ============ build mask for qc-cases ============
                qc_mask = tf.where((else_mask + for_mask) >= 2, tf.ones_like(self.qt), tf.zeros_like(self.qt))

                # ============ build mask for min-cases ============
                min_mask = tf.where((qt_mask + for_mask) < 1, tf.ones_like(self.qt), tf.zeros_like(self.qt))

                # ============ combine to pseudo q-function ============
                q_out = tf.multiply(self.qt, qt_mask) + \
                        tf.multiply(self.qc, qc_mask) + \
                        min_mask * - 5.0
            else: #todo vor dem veröffentlichen auseinanderwursteln, das hier wird genutzt für linear scalarization
                action_scores = tf_layers.fully_connected(q_united_out, num_outputs=self.n_actions, activation_fn=None)
                if self.dueling:
                    with tf.variable_scope("state_value"):
                        state_out = extracted_features
                        for layer_size in layers:
                            state_out = tf_layers.fully_connected(state_out, num_outputs=layer_size, activation_fn=None)
                            if layer_norm:
                                state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
                            state_out = act_fun(state_out)
                        state_score = tf_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                    action_scores_mean = tf.reduce_mean(action_scores, axis=1)
                    action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
                    q_out = state_score + action_scores_centered
                else:
                    q_out = action_scores
        self.q_values = q_out
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        if self.generalizedTLO:
            q_values, qt, qc, thresh, actions_proba, state_vals = self.sess.run([self.q_values, self.qt, self.qc,
                                                                                 self.thresh, self.policy_proba,
                                                                                 self.state_vals], {self.obs_ph: obs})
        else:
            q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})

        # print(f"{raw_q}, {thresh} => {q_values}")
        if deterministic:
            actions = np.argmax(q_values, axis=1)
            # print(f'{obs.flatten()} q-vals: {q_values}, a: {actions}')
            # if graphical-state:
            # print(f'{obs[0,-1,:2,:].flatten()} q-vals: {q_values}, a: {actions}')
        else:
            # Unefficient sampling
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])
        if self.generalizedTLO:
            return actions, {'q':q_values, 'qt':qt, 'qc':qc,'thr':thresh}, None
        else:
            return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})


class CnnPolicyExtendedDST(GtloPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=True, generalizedTLO=False, **_kwargs):
        self.generalizedTLO = generalizedTLO
        super(CnnPolicyExtendedDST, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                                   feature_extraction="cnn", cnn_extractor=gtlo_dst,
                                                   obs_phs=obs_phs, dueling=dueling,
                                                   layer_norm=False, **_kwargs)


class CnnPolicyExtendedDeepDrawing(GtloPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=True, generalizedTLO=False, **_kwargs):
        self.generalizedTLO = generalizedTLO
        super(CnnPolicyExtendedDeepDrawing, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                                           feature_extraction="cnn", cnn_extractor=gtlo_deepdrawing2rts,
                                                           obs_phs=obs_phs, dueling=dueling,
                                                           layer_norm=False, **_kwargs)


class LnMlpPolicyExtended(GtloPolicy):
    """
    Policy object that implements DQN policy, using a MLP (2 layers of 64), with layer normalisation

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=True, generalizedTLO=False, **_kwargs):
        self.generalizedTLO = generalizedTLO
        super(LnMlpPolicyExtended, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                                  feature_extraction="mlp", obs_phs=obs_phs,
                                                  layer_norm=True, dueling=True, **_kwargs)


register_policy("gTLOdst", CnnPolicyExtendedDST)
register_policy("gTLOdeepdrawing", CnnPolicyExtendedDeepDrawing)


# register_policy("LnMlpPolicyExtended", LnMlpPolicyExtended)

class MultibatchDQN(DQN):
    def __init__(self, *args, batches_per_training=1, q_val_ret=False, generalizedTLO=False, **kwargs):
        self.batches_per_training = batches_per_training
        self.q_val_ret = q_val_ret
        self.generalizedTLOModel = generalizedTLO
        super().__init__(*args, **kwargs)

    def setup_model(self):
        if not self.generalizedTLOModel:
            super().setup_model()
        else: # have to overwrite this part, so the costumized (imported) build_train function is used
            with SetVerbosity(self.verbose):
                assert not isinstance(self.action_space, gym.spaces.Box), \
                    "Error: DQN cannot output a gym.spaces.Box action space."

                # If the policy is wrap in functool.partial (e.g. to disable dueling)
                # unwrap it to check the class type
                if isinstance(self.policy, partial):
                    test_policy = self.policy.func
                else:
                    test_policy = self.policy
                assert issubclass(test_policy, DQNPolicy), "Error: the input policy for the DQN model must be " \
                                                           "an instance of DQNPolicy."

                self.graph = tf.Graph()
                with self.graph.as_default():
                    self.set_random_seed(self.seed)
                    self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

                    self.act, self._train_step, self.update_target, self.step_model = build_train(
                        q_func=partial(self.policy, **self.policy_kwargs),
                        ob_space=self.observation_space,
                        ac_space=self.action_space,
                        optimizer=optimizer,
                        gamma=self.gamma,
                        grad_norm_clipping=10,
                        param_noise=self.param_noise,
                        sess=self.sess,
                        full_tensorboard_log=self.full_tensorboard_log,
                        double_q=self.double_q
                    )
                    self.proba_step = self.step_model.proba_step
                    self.params = tf_util.get_trainable_vars("deepq")

                    # Initialize the parameters and copy them to the target network.
                    tf_util.initialize(self.sess)
                    self.update_target(sess=self.sess)

                    self.summary = tf.summary.merge_all()

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            # Create the replay buffer
            if self.prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = total_timesteps
                else:
                    prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                    initial_p=self.prioritized_replay_beta0,
                                                    final_p=1.0)
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.beta_schedule = None

            if replay_wrapper is not None:
                assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                self.replay_buffer = replay_wrapper(self.replay_buffer)

            # Create the schedule for exploration starting from 1.
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                              initial_p=self.exploration_initial_eps,
                                              final_p=self.exploration_final_eps)

            episode_rewards = [0.0]
            episode_successes = []

            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()

            reset = True
            obs = self.env.reset()
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                obs_ = self._vec_normalize_env.get_original_obs().squeeze()

            for _ in range(total_timesteps):
                # Take action and update exploration to the newest value
                kwargs = {}
                if not self.param_noise:
                    update_eps = self.exploration.value(self.num_timesteps)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = \
                        -np.log(1. - self.exploration.value(self.num_timesteps) +
                                self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True
                with self.sess.as_default():
                    action, q = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)
                    if self.q_val_ret:
                        action = action[0]

                env_action = action
                reset = False
                new_obs, rew, done, info = self.env.step(env_action)

                self.num_timesteps += 1

                # Stop training if return value is False
                callback.update_locals(locals())
                if callback.on_step() is False:
                    break

                # Store only the unnormalized version
                if self._vec_normalize_env is not None:
                    new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                    reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                else:
                    # Avoid changing the original ones
                    obs_, new_obs_, reward_ = obs, new_obs, rew
                # Store transition in the replay buffer.
                self.replay_buffer_add(obs_, action, reward_, new_obs_, done, info)
                obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    obs_ = new_obs_

                if writer is not None:
                    ep_rew = np.array([reward_]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    tf_util.total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                                                        self.num_timesteps)

                episode_rewards[-1] += reward_
                if done:
                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)
                    reset = True

                # Do not train if the warmup phase is not over
                # or if there are not enough samples in the replay buffer
                can_sample = self.replay_buffer.can_sample(self.batch_size)
                if can_sample and self.num_timesteps > self.learning_starts \
                        and self.num_timesteps % self.train_freq == 0:

                    callback.on_rollout_end()
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    # pytype:disable=bad-unpacking
                    for i in range(self.batches_per_training):
                        if self.prioritized_replay:
                            assert self.beta_schedule is not None, \
                                "BUG: should be LinearSchedule when self.prioritized_replay True"
                            experience = self.replay_buffer.sample(self.batch_size,
                                                                   beta=self.beta_schedule.value(self.num_timesteps),
                                                                   env=self._vec_normalize_env)
                            (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                        else:
                            obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size,
                                                                                                    env=self._vec_normalize_env)
                            weights, batch_idxes = np.ones_like(rewards), None
                            # weights = weights.flatten()
                        # pytype:enable=bad-unpacking

                        if writer is not None:
                            # run loss backprop with summary, but once every 100 steps save the metadata
                            # (memory, compute time, ...)
                            if (1 + self.num_timesteps) % 100 == 0:
                                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                run_metadata = tf.RunMetadata()
                                summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                      dones, weights, sess=self.sess,
                                                                      options=run_options,
                                                                      run_metadata=run_metadata)
                                if i == 1:
                                    writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                            else:
                                summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                      dones, weights, sess=self.sess)
                            writer.add_summary(summary, self.num_timesteps)
                        else:
                            summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                  dones, weights,
                                                                  sess=self.sess)

                        if self.prioritized_replay:
                            if self.generalizedTLOModel:
                                new_priorities = np.max(np.abs(td_errors), axis=1) + self.prioritized_replay_eps
                            else:
                                new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                            assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
                            self.replay_buffer.update_priorities(batch_idxes, new_priorities)

                    callback.on_rollout_start()

                if can_sample and self.num_timesteps > self.learning_starts and \
                        self.num_timesteps % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.update_target(sess=self.sess)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_100ep_reward = -np.inf
                else:
                    mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    logger.record_tabular("steps", self.num_timesteps)
                    logger.record_tabular("episodes", num_episodes)
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("% time spent exploring",
                                          int(100 * self.exploration.value(self.num_timesteps)))
                    logger.dump_tabular()

        callback.on_training_end()
        return self

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.sess.as_default():
            actions, q_vals, _ = self.step_model.step(observation, deterministic=deterministic)

        if not vectorized_env:
            actions = actions[0]

        return actions, q_vals
