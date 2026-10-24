import os

import numpy as np


# Design-choice experiments from the paper (Table 3 action spaces, Table 4 time counters and observation stacking).
# Every variant starts from the basic set-up: only the case study's own state variables, no time counters, no
# stacked observations and no reward penalties, and then changes one thing. 'tuned' (the default) keeps a run
# script's own settings unchanged.
BASIC_STATE_VARS = {
    'al4_bat_ea': ['pool_price'],
    'de1_bat_hes': ['pool_price', 're_power'],
}
TIME_FEATURES = {
    'time_regular': ['h', 'week', 'month'],
    'time_sincos': ['cos_h', 'sin_h', 'cos_w', 'sin_w', 'cos_m', 'sin_m'],
}
FRAME_STACK = {
    'stack2': 2,
    'stack4': 4,
}
N_DISCRETE_ACTIONS = {
    'disc3': 3,
    'disc17': 17,
}
VARIANTS = ['tuned', 'basic', *TIME_FEATURES, *FRAME_STACK, *N_DISCRETE_ACTIONS]


def apply_variant(env_kwargs: dict, exp_params: dict, discrete_actions):
    """Apply the variant named by the VARIANT environment variable. Returns the discrete actions to use."""
    variant = os.environ.get('VARIANT', 'tuned')
    assert variant in VARIANTS, f"Unknown VARIANT '{variant}', use one of {VARIANTS}"
    if variant == 'tuned':
        return discrete_actions

    env_kwargs['state_vars'] = BASIC_STATE_VARS[env_kwargs['env_name']] + TIME_FEATURES.get(variant, [])
    exp_params['inactive_penalty'] = None
    exp_params['action_corr_penalty'] = None
    exp_params['soc_penalty'] = None
    exp_params['frame_stack'] = FRAME_STACK.get(variant)

    if variant in N_DISCRETE_ACTIONS:
        discrete_actions = [np.array([i]) for i in np.linspace(-1, 1, N_DISCRETE_ACTIONS[variant])]
    return discrete_actions
