import os
import re

import numpy as np

from utils.scheduler import linear_schedule


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

# Our own hyperparameter experiments (Milestone 2). Unlike the variants above, these keep the tuned set-up and
# change one hyperparameter, e.g. VARIANT=gamma0.99, explore0.3 (DQN's exploration_fraction), ent0.1 (PPO's
# ent_coef) or lr_x3 (3 times the tuned learning rate; a linear schedule keeps decaying from the new start).
HPARAMS = {
    'gamma': 'gamma',
    'explore': 'exploration_fraction',
    'ent': 'ent_coef',
    'lr_x': 'learning_rate',
}
HPARAM_VARIANT = re.compile(r'(gamma|explore|ent|lr_x)(\d+(?:\.\d+)?)')


def apply_variant(env_kwargs: dict, exp_params: dict, discrete_actions):
    """Apply the variant named by the VARIANT environment variable. Returns the discrete actions to use."""
    variant = os.environ.get('VARIANT', 'tuned')
    is_hparam = HPARAM_VARIANT.fullmatch(variant) is not None
    assert variant in VARIANTS or is_hparam, \
        f"Unknown VARIANT '{variant}', use one of {VARIANTS} or {list(HPARAMS)} followed by a value"
    if variant == 'tuned' or is_hparam:  # hyperparameter variants are applied by apply_hparams()
        return discrete_actions

    env_kwargs['state_vars'] = BASIC_STATE_VARS[env_kwargs['env_name']] + TIME_FEATURES.get(variant, [])
    exp_params['inactive_penalty'] = None
    exp_params['action_corr_penalty'] = None
    exp_params['soc_penalty'] = None
    exp_params['frame_stack'] = FRAME_STACK.get(variant)

    if variant in N_DISCRETE_ACTIONS:
        discrete_actions = [np.array([i]) for i in np.linspace(-1, 1, N_DISCRETE_ACTIONS[variant])]
    return discrete_actions


def apply_hparams(rl_params: dict, exp_params: dict):
    """Apply a hyperparameter variant (see HPARAMS) named by the VARIANT environment variable, if any."""
    match = HPARAM_VARIANT.fullmatch(os.environ.get('VARIANT', 'tuned'))
    if match is None:
        return
    name, value = match.group(1), float(match.group(2))
    key = HPARAMS[name]
    assert key in rl_params, f"VARIANT {match.group(0)} changes '{key}', which this algorithm doesn't have"

    def show(v):  # starting value of a linear schedule, or the value itself
        v = v(1) if callable(v) else v
        return f'{v:g}' if isinstance(v, (int, float)) else str(v)

    old = rl_params[key]
    if name == 'lr_x':
        rl_params[key] = linear_schedule(value * old(1)) if callable(old) else value * old
    else:
        rl_params[key] = value
    change = f"{key}: {show(old)} -> {show(rl_params[key])}"
    exp_params['changed'] = change  # saved in inputs.json
    print(f"VARIANT {match.group(0)}, {change}")
