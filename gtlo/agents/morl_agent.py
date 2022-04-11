import numpy as np
from pathlib import Path
import gym
from gym import logger
from datetime import datetime
import os
import shutil
import configparser
import json
from fruit.envs.juice import FruitEnvironment
from fruit.state.processor import AtariProcessor
import logging
import argparse

from gtlo.fruityGym import DynamicConfigurationWrapper, NamedDeepSeaTreasure
from gtlo.sb_costumizations import CnnPolicyExtendedDST, CnnPolicyExtendedDeepDrawing, MultibatchDQN
from gtlo.sb_callbacks import MORLEvaluation, RandomConfigurationSetter, MORLEvaluationStochastic
from gtlo.helpers import LinearScalarizer, RewardScaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gTLO')
    parser.add_argument('--config', type=str,
                        help='path to the config (ini) file')
    args = parser.parse_args()
    if args.config:
        config_path = args.config
    else:
        config_path = './morl_config.ini'

    logger.set_level(logger.INFO)
    logger = logging.getLogger('')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # ============================ read in config ============================
    config = configparser.ConfigParser()
    morl_config = Path(config_path)
    config.read(morl_config)

    general_config = config['general parameters']

    # general parameters
    experiment_label = general_config.get('experiment_label')
    env_id = general_config.get('env_id')
    agent_type = general_config.get('agent_type')

    experiment_count = general_config.getint('run_count')
    training_steps = general_config.getint('steps_count')
    eval_freq = general_config.getint('eval_frequency')

    # multirun folder
    exp_path = general_config.get('experiment_storage')
    if env_id == '2d-deepdrawing-5ts-stressoffsetstate-v0':
        env_alias = '2d_dd_v0'
        env_type = 'deepdrawing'

        from gtlo.morl_gym_fem import DeepDrawingMORLWrapper
        from gym_fem.fem_env import FEMCSVLogger

    elif env_id == 'fruity-gym-v0':
        dst_config = config['DST parameters']
        env_alias = dst_config.get('game_id')
        env_type = 'dst'
    else:
        env_alias = env_id
        env_type = None

    exp_name = f'{env_alias}_{agent_type}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}_{experiment_label}'
    multirun_dir = Path(f'{exp_path}/{exp_name}')
    multirun_dir.mkdir()

    # copy agent file and config to experiment folder
    shutil.copy(Path(os.path.dirname(os.path.abspath(__file__))).joinpath('morl_agent.py'),
                multirun_dir.joinpath('morl_agent.py'))
    shutil.copy(Path(config_path),
                multirun_dir.joinpath('config.ini'))

    morl_config = config['morl parameters']
    hypervolume_ref = json.loads(morl_config.get('hypervolume_ref'))
    weights_type = morl_config.get('weights_type')
    outer_loop = morl_config.getboolean('outer_loop')
    weights_count = morl_config.getint('weights_count')

    run_count = experiment_count
    if outer_loop:
        run_count *= weights_count

    # main experiment loop
    for experiment_i in range(run_count):
        if outer_loop:
            n_loops = experiment_i // weights_count
            weights_i = experiment_i % weights_count
            run_name = f'{n_loops}/{weights_type}_{weights_i}_{env_alias}_{agent_type}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}'
        else:
            run_name = f'{env_alias}_{agent_type}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}'

        outdir = multirun_dir.joinpath(f'{run_name}_{experiment_i}')
        outdir.mkdir(exist_ok=True, parents=True)

        # ======================================= create env ===============================================================

        if env_type == 'deepdrawing':
            dd_config = config['Deep Drawing parameters']
            r_terms = json.loads(dd_config.get('reward_terms'))
            env = gym.make(env_id)
            env = DeepDrawingMORLWrapper(env, r_terms=r_terms,
                                         prohibit_negative_rewards=True)
            env = FEMCSVLogger(env, outdir=outdir)
        elif env_type == 'dst':
            game_id = dst_config.get('game_id')
            max_episode_steps = dst_config.getint('max_episode_steps')
            render = dst_config.getboolean('render')
            transition_noise = dst_config.getfloat('transition_noise', fallback=0)
            n_eval_episodes = dst_config.getint('n_eval_episodes', fallback=1)
            if game_id == 'van2014multi':
                dst_game = NamedDeepSeaTreasure(graphical_state=True, width=10, seed=100, render=render,
                                                max_treasure=124, speed=1000, transition_noise=transition_noise,
                                                min_depth=1, min_vertical_step=0, named_game='van2014multi')
                eval_dst_game = NamedDeepSeaTreasure(graphical_state=True, width=10, seed=100, render=render,
                                                     max_treasure=124, speed=1000, transition_noise=transition_noise,
                                                     min_depth=1, min_vertical_step=0, named_game='van2014multi')

                fruit_env = FruitEnvironment(dst_game, max_episode_steps=max_episode_steps - 1,
                                             state_processor=AtariProcessor())
                env = gym.make('fruity-gym-v0', fruit_env=fruit_env)


        else:
            raise NotImplementedError(f'environment {env_id} not known')

        # ======================================= create agent =============================================================
        # dqn parameters
        dqn_config = config['dqn parameters']
        dqn_callbacks = []
        batches_per_training = dqn_config.getint('dqn_batches')
        buffer_size = dqn_config.getint('buffer_size')
        gamma = dqn_config.getfloat('gamma')
        target_network_update_freq = dqn_config.getint('target_network_update_freq')
        extension_priorreplay = dqn_config.getboolean('extension_priorreplay')
        learning_starts = dqn_config.getint('learning_starts', fallback=1000)

        reward_scale = morl_config.getfloat('reward_scale', fallback=1)
        if agent_type == 'gTLO':
            reward_manipulator = RewardScaler(reward_scale)
            # gtlo_config = config['gtlo parameters']
            # gtlo parameters
        elif agent_type == 'linear_dqn':
            if env_type == 'dst' and game_id == 'van2014multi':
                r_min = np.array([0, -max_episode_steps])
                r_max = np.array([124, -1])
            elif env_type == 'deepdrawing' and env_alias == '2d_dd_v0':
                r_min = np.array([0,0])
                r_max = np.array([1.2, 1.2])
            else:
                raise NotImplementedError()

            reward_manipulator = LinearScalarizer(min_r=r_min, max_r=r_max, min_max_scaling=True,
                                                  buffer_rewards=True)
        else:
            raise NotImplementedError(f'agent type {agent_type} not known')

        if weights_type == 'eql':
            weights_min = morl_config.getfloat('weights_min')
            weights_max = morl_config.getfloat('weights_max')
            step = (weights_max - weights_min) / (weights_count - 1)
            weights = np.arange(weights_min, weights_max + step, step)
            configurations = np.hstack([weights.reshape((-1, 1)),
                                        weights_max - weights.reshape((-1, 1))
                                        ])
        elif weights_type == 'exact' and env_type == 'dst':
            t_confs = []
            treasure_vals = dst_game.get_treasure()
            t_confs.append((treasure_vals[0]) / 2)
            for i in range(1, len(treasure_vals)):
                t_confs.append((treasure_vals[i - 1] + treasure_vals[i]) / 2)
            configurations = np.ones([len(t_confs), 2])
            t_confs = np.array(t_confs) * .1
            configurations[:, 0] = t_confs
        else:
            raise NotImplementedError(f'weights type {weights_type} not implemented for env')

        if outer_loop:
            configurations = np.array([configurations[weights_i]])

        # todo refactor this part
        scal_wrapper = DynamicConfigurationWrapper(env, configurations[0], reward_manipulator=reward_manipulator,
                                                   manipulate_state=True, configurations=configurations,
                                                   step_state=False)
        env = scal_wrapper

        genTLO = (agent_type == 'gTLO')
        policy_kwargs = {'dueling': False, 'generalizedTLO': genTLO}

        if env_type == 'deepdrawing':
            policy = CnnPolicyExtendedDeepDrawing
        elif env_type == 'dst':
            policy = CnnPolicyExtendedDST

        dqn_kwargs = {'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.0,
                      'exploration_fraction': 0.9, 'prioritized_replay': extension_priorreplay,
                      'gamma': gamma, 'double_q': False, 'buffer_size': buffer_size,
                      'target_network_update_freq': target_network_update_freq
                      }

        dqn_model = MultibatchDQN(policy, env, verbose=1, tensorboard_log=Path(outdir).joinpath('tensorboard_log'),
                                  full_tensorboard_log=True, policy_kwargs=policy_kwargs,
                                  batches_per_training=batches_per_training, generalizedTLO=genTLO,
                                  learning_starts=learning_starts, **dqn_kwargs)

        # ======================================= create evaluation ========================================================
        dynscal = RandomConfigurationSetter(configurations, scal_wrapper)
        dqn_callbacks.append(dynscal)

        if env_type == 'dst':
            eval_fruit_env = FruitEnvironment(eval_dst_game, max_episode_steps=max_episode_steps - 1,
                                              state_processor=AtariProcessor())
            eval_env = gym.make('fruity-gym-v0', fruit_env=eval_fruit_env)
            eval_env = DynamicConfigurationWrapper(eval_env, configurations[0], reward_manipulator=None,
                                                   manipulate_state=True, step_state=False)

            morleval = MORLEvaluation(configurations=configurations, eval_env=eval_env, n_eval_episodes=n_eval_episodes,
                                      manipulator=reward_manipulator,
                                      eval_freq=eval_freq, hypervolume_ref_point=hypervolume_ref,
                                      logdir=outdir)
        elif env_type == 'deepdrawing':
            eval_env = gym.make(env_id)
            eval_env = DeepDrawingMORLWrapper(eval_env, r_terms=r_terms)
            eval_log_outdir = outdir.joinpath('eval_logs')
            eval_log_outdir.mkdir()
            eval_env = FEMCSVLogger(eval_env, outdir=eval_log_outdir)
            eval_env = DynamicConfigurationWrapper(eval_env, configurations[0], reward_manipulator=None,
                                                   manipulate_state=True, step_state=False)

            morleval = MORLEvaluationStochastic(eval_frictions=[0.014 * i for i in range(1, 10)],
                                                configurations=configurations, eval_env=eval_env,
                                                n_eval_episodes=1, manipulator=reward_manipulator,
                                                eval_freq=eval_freq, hypervolume_ref_point=hypervolume_ref,
                                                logdir=outdir)
        dqn_callbacks.append(morleval)

        # ======================================= run agent ================================================================
        dqn_model.learn(training_steps, callback=dqn_callbacks)
        dqn_model.save(outdir.joinpath('model'))
