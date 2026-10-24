import os
import json
from typing import Any

from stable_baselines3.common.noise import NormalActionNoise

from environments.environment import *
from environments.env_params import *
from utils.net_design import activation_fn_dict, net_arch_dict
from utils.scheduler import linear_schedule
from utils.utils import create_stats_file
from train.train import train_rl_agent


# CHOOSE ENVIRONMENT/CASE STUDY
# -----------------------------------------------------------------
# DE-LF/REU
ENV = HES_BESS_Env1
ENV_KWARGS = de1_bat_hes
ENV_KWARGS['state_vars'] = ['pool_price', 're_power', 'cos_h', 'sin_h', 'cos_w', 'sin_w', 'cos_m', 'sin_m']

# LOG
CREATE_LOG = True
VERBOSE = 0
LOGGER_TYPE = ["csv", "tensorboard"]
SAVE_PATH = os.path.join('log/', ENV_KWARGS['env_name'], 'ddpg/run', input('Save in folder: ')) \
    if CREATE_LOG else None

# The paper doesn't report tuned DDPG hyperparameters, so this setup reuses the tuned SAC setup for DE-LF/REU
# (Table C.8 and 2 stacked observations: same experiment design, replay and update settings, network) and
# adds Gaussian exploration noise, which DDPG's deterministic policy needs to explore. NOISE_SIGMA sets the
# noise level (0.1 scored slightly higher than 0.3 on DE-LF/REU, within one std).
NOISE_SIGMA = float(os.environ.get('NOISE_SIGMA', 0.1))

# EXP PARAMS
EXP_PARAMS = {
    'n_runs': int(os.environ.get('N_RUNS', 5)),
    'n_episodes': int(os.environ.get('N_EPISODES', 50)),
    'seed': 22,
    # Reward wrapper
    'inactive_penalty': None,
    'action_corr_penalty': None,
    'soc_penalty': 200,
    # Env
    'flatten_obs': True,
    'frame_stack': 2,
    # Normalization
    'norm_obs': True,
    'norm_reward': True,
    # Evaluation
    'eval_while_training': True,
    'eval_freq': 8760 * 1,
}

# DDPG PARAMS
RL_PARAMS: dict[str, Any] = {
    'policy': "MlpPolicy" if EXP_PARAMS['flatten_obs'] else 'MultiInputPolicy',
    'learning_rate': 1e-4,  # Default: 1e-3
    'buffer_size': 500_000,  # Default: 1e6
    'learning_starts': 1000,  # Default: 100
    'batch_size': 512,  # Default: 100
    'tau': 0.006985,  # Default: 0.005
    'gamma': 0.999,  # Default: 0.99
    'train_freq': 35,  # Default: (1, 'episode')
    'gradient_steps': -1,  # Default: -1
    'action_noise': NormalActionNoise(mean=np.zeros(1), sigma=NOISE_SIGMA * np.ones(1)),  # Default: None

    'policy_kwargs': {
        'net_arch': 'extra_large',  # Default: None
        'activation_fn': 'tanh',  # Default: relu
        'n_critics': 2,  # Default: 1
        'share_features_extractor': True,  # Default: False
    }
}

if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'EXP_PARAMS': EXP_PARAMS,
            'RL_PARAMS': str(RL_PARAMS),
            'PLANT_PARAMS': ENV_KWARGS,
        }, f)

RL_PARAMS['policy_kwargs']['net_arch'] = net_arch_dict[RL_PARAMS['policy_kwargs']['net_arch']]
RL_PARAMS['policy_kwargs']['activation_fn'] = activation_fn_dict[RL_PARAMS['policy_kwargs']['activation_fn']]

# RUN set: train only that seed (one Slurm array task per seed)
RUNS = [int(os.environ['RUN'])] if 'RUN' in os.environ else range(EXP_PARAMS['n_runs'])

for run in RUNS:
    train_rl_agent(
        agent='ddpg',
        run=run,
        path=SAVE_PATH,
        exp_params=EXP_PARAMS,
        env_id=ENV,
        env_kwargs=ENV_KWARGS,
        discrete_actions=None,
        rl_params=RL_PARAMS,
        verbose=VERBOSE,
        logger_type=LOGGER_TYPE,
    )

# GET STATISTICS FROM MULTIPLE RUNS
if CREATE_LOG and EXP_PARAMS['n_runs'] > 1 and 'RUN' not in os.environ:
    create_stats_file(SAVE_PATH, EXP_PARAMS)
