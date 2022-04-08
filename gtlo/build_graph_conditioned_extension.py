import tensorflow as tf
from gym.spaces import MultiDiscrete

from stable_baselines.common import tf_util

""" 
qt: expected max treasure-reward per action
qc: expected steps conditioned on treasure per action
ct: treasure-condition (encoded in state)
ga: greedy action

if qc < ct:
    ga = argmax(qt)
else:
    ga = argmax(qc)    
"""


def build_act(q_func, ob_space, ac_space, stochastic_ph, update_eps_ph, sess):
    """
    Creates the act function:

    :param q_func: (DQNPolicy) the policy
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param stochastic_ph: (TensorFlow Tensor) the stochastic placeholder
    :param update_eps_ph: (TensorFlow Tensor) the update_eps placeholder
    :param sess: (TensorFlow session) The current TensorFlow session
    :return: (function (TensorFlow Tensor, bool, float): TensorFlow Tensor, (TensorFlow Tensor, TensorFlow Tensor)
        act function to select and action given observation (See the top of the file for details),
        A tuple containing the observation placeholder and the processed observation placeholder respectively.
    """
    eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

    policy = q_func(sess, ob_space, ac_space, 1, 1, None)
    obs_phs = (policy.obs_ph, policy.processed_obs)
    deterministic_actions = tf.argmax(policy.q_values, axis=1)

    batch_size = tf.shape(policy.obs_ph)[0]
    n_actions = ac_space.nvec if isinstance(ac_space, MultiDiscrete) else ac_space.n
    random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=n_actions, dtype=tf.int64)
    chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
    stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

    output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
    update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
    _act = tf_util.function(inputs=[policy.obs_ph, stochastic_ph, update_eps_ph],
                            outputs=(output_actions, policy.q_values),
                            givens={update_eps_ph: -1.0, stochastic_ph: True},
                            updates=[update_eps_expr])

    def act(obs, stochastic=True, update_eps=-1):
        return _act(obs, stochastic, update_eps)

    return act, obs_phs


# ignore the param-noise case
def build_act_with_param_noise(q_func, ob_space, ac_space, stochastic_ph, update_eps_ph, sess,
                               param_noise_filter_func=None):
    raise NotImplementedError


def build_train(q_func, ob_space, ac_space, optimizer, sess, grad_norm_clipping=None,
                gamma=1.0, double_q=True, scope="deepq", reuse=None,
                param_noise=False, param_noise_filter_func=None, full_tensorboard_log=False):
    """
    Creates the train function:

    :param q_func: (DQNPolicy) the policy
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param reuse: (bool) whether or not to reuse the graph variables
    :param optimizer: (tf.train.Optimizer) optimizer to use for the Q-learning objective.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param grad_norm_clipping: (float) clip gradient norms to this value. If None no clipping is performed.
    :param gamma: (float) discount rate.
    :param double_q: (bool) if true will use Double Q Learning (https://arxiv.org/abs/1509.06461). In general it is a
        good idea to keep it enabled.
    :param scope: (str or VariableScope) optional scope for variable_scope.
    :param reuse: (bool) whether or not the variables should be reused. To be able to reuse the scope must be given.
    :param param_noise: (bool) whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    :param param_noise_filter_func: (function (TensorFlow Tensor): bool) function that decides whether or not a
        variable should be perturbed. Only applicable if param_noise is True. If set to None, default_param_noise_filter
        is used by default.
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly

    :return: (tuple)

        act: (function (TensorFlow Tensor, bool, float): TensorFlow Tensor) function to select and action given
            observation. See the top of the file for details.
        train: (function (Any, numpy float, numpy float, Any, numpy bool, numpy float): numpy float)
            optimize the error in Bellman's equation. See the top of the file for details.
        update_target: (function) copy the parameters from optimized Q function to the target Q function.
            See the top of the file for details.
        step_model: (DQNPolicy) Policy for evaluation
    """
    n_actions = ac_space.nvec if isinstance(ac_space, MultiDiscrete) else ac_space.n
    with tf.variable_scope("input", reuse=reuse):
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

    with tf.variable_scope(scope, reuse=reuse):
        if param_noise:
            act_f, obs_phs = build_act_with_param_noise(q_func, ob_space, ac_space, stochastic_ph, update_eps_ph, sess,
                                                        param_noise_filter_func=param_noise_filter_func)
        else:
            act_f, obs_phs = build_act(q_func, ob_space, ac_space, stochastic_ph, update_eps_ph, sess)

        # q network evaluation
        with tf.variable_scope("step_model", reuse=True, custom_getter=tf_util.outer_scope_getter("step_model")):
            step_model = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=True, obs_phs=obs_phs)
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/model")
        # target q network evaluation
        with tf.variable_scope("target_q_func", reuse=False):
            target_policy = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=False)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope=tf.get_variable_scope().name + "/target_q_func")

        # compute estimate of best possible value starting from state at t + 1
        double_obs_ph = target_policy.obs_ph

    with tf.variable_scope("loss", reuse=reuse):
        generalizedTLO = hasattr(step_model, 'qt')

        # set up placeholders
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        if generalizedTLO:
            rew_t_ph = tf.placeholder(tf.float32, [None, 2], name="reward")
            importance_weights_ph = tf.placeholder(tf.float32, [None, 2], name="weight")
        else:
            rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
            importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # from here on everything has to be done  per objective
        # q scores for actions which we know were selected in the given state.
        if generalizedTLO:
            qt = step_model.qt
            t_thresh = step_model.thresh[:, 0]
            print(f"t_thresh.shape {step_model.thresh.shape} --- {step_model.thresh} // {t_thresh.shape} {t_thresh}")
            qc = step_model.qc

            q_t_selected = tf.concat([tf.reduce_sum(qt * tf.one_hot(act_t_ph, n_actions), axis=1),
                                      tf.reduce_sum(qc * tf.one_hot(act_t_ph, n_actions), axis=1)], axis=0)
            q_t_selected = tf.transpose(tf.reshape(q_t_selected, [2, -1]))
        else:
            q_t_selected = tf.reduce_sum(step_model.q_values * tf.one_hot(act_t_ph, n_actions), axis=1)

        # compute estimate of best possible value starting from state at t + 1
        if generalizedTLO:
            # restrict to actions with qt > thresh or (if none) with max qt
            # (1) maximum thresholded values in tp1
            qt_tp1_clipped = tf.clip_by_value(target_policy.qt,
                                          clip_value_min=-999,
                                          clip_value_max=tf.reshape(t_thresh, [-1, 1]))
            max_qt_tp1 = tf.reduce_max(qt_tp1_clipped, axis=1)
            # (2) mask maximum thresholded values
            thresh_mask = tf.equal(qt_tp1_clipped, tf.reshape(max_qt_tp1, [-1, 1]))
            # (3) build matrix of qc values with very low values where the threshold constraint is not met
            # (by using the mask)
            # todo not very elegant (assumes that max qc is >-100)
            qc_tp1_masked = (tf.cast(thresh_mask, tf.float32) * 100) - 100 + target_policy.qc
            # (4) build tp1 q-vectors for off-policy updates:
            # for qt: standard q-learning update (no constraints)
            # for qc: by using the created masked qc matrix instead
            q_tp1_best_masked = tf.concat([(1.0 - done_mask_ph) * tf.reduce_max(target_policy.qt, axis=1),
                                           (1.0 - done_mask_ph) * tf.reduce_max(qc_tp1_masked, axis=1)],
                                          axis=0)
            q_tp1_best_masked = tf.transpose(tf.reshape(q_tp1_best_masked, [2, -1]))
        else:
            q_tp1_best = tf.reduce_max(target_policy.q_values, axis=1)
            q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        if generalizedTLO:
            q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked
            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
            errors_t = tf_util.huber_loss(td_error[:, 0])
            errors_c = tf_util.huber_loss(td_error[:, 1])
            weighted_error = tf.reduce_sum([tf.reduce_mean(importance_weights_ph[:, 0] * errors_t),
                                            tf.reduce_mean(importance_weights_ph[:, 1] * errors_c)])
        else:
            # compute RHS of bellman equation
            q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked
            # compute the error (potentially clipped)
            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
            errors = tf_util.huber_loss(td_error)
            weighted_error = tf.reduce_mean(importance_weights_ph * errors)

        tf.summary.scalar("loss", weighted_error)

        if full_tensorboard_log:
            tf.summary.histogram("td_error", td_error)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # compute optimization op (potentially with gradient clipping)
        gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
        if grad_norm_clipping is not None:
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)

    with tf.variable_scope("input_info", reuse=False):
        tf.summary.scalar('rewards', tf.reduce_mean(rew_t_ph))
        tf.summary.scalar('importance_weights', tf.reduce_mean(importance_weights_ph))

        if full_tensorboard_log:
            tf.summary.histogram('rewards', rew_t_ph)
            tf.summary.histogram('importance_weights', importance_weights_ph)
            if tf_util.is_image(obs_phs[0]):
                tf.summary.image('observation', obs_phs[0])
            elif len(obs_phs[0].shape) == 1:
                tf.summary.histogram('observation', obs_phs[0])

    optimize_expr = optimizer.apply_gradients(gradients)

    summary = tf.summary.merge_all()

    # Create callable functions
    train = tf_util.function(
        inputs=[
            obs_phs[0],
            act_t_ph,
            rew_t_ph,
            target_policy.obs_ph,
            double_obs_ph,
            done_mask_ph,
            importance_weights_ph
        ],
        # used for debug purposes
        # outputs=[summary, td_error, t_thresh, target_policy.qt, target_policy.qc, qt_clipped, max_qt, thresh_mask, qc_tp1_masked, q_tp1_best_masked, q_t_selected_target],
        outputs=[summary, td_error],
        updates=[optimize_expr]
    )
    update_target = tf_util.function([], [], updates=[update_target_expr])

    return act_f, train, update_target, step_model
