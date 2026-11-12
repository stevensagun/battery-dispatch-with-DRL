import os
import sys
import statistics

from main.summarize_results import ENV_NAMES, seed_rewards


# Prints our hyperparameter experiments (hpc/submit_hparams.sh) next to the tuned runs (run/full): mean ± std
# across seeds of the average reward over the last 5 training episodes, and of the final evaluation, in thousand
# C$. A row shows (n/5) when only n seeds have finished.
# Usage: python -m main.summarize_hparams <delf|abea>
TASK = sys.argv[1]
BASE = os.path.join('log', ENV_NAMES[TASK])

# Tuned values from the run scripts (paper, Appendix C)
TUNED = {
    'abea': {'gamma': '0.950', 'explore': '0.1', 'ent': '0.249'},
    'delf': {'gamma': '0.975', 'explore': '0.5', 'ent': '0.970'},
}[TASK]

# (title, algorithm, [(label, results folder)]); 'full' is the tuned run
EXPERIMENTS = [
    ('DQN discount factor', 'dqn',
     [('0.9', 'gamma0.9'), ('0.95', 'gamma0.95'), ('0.99', 'gamma0.99'), ('0.999', 'gamma0.999'),
      (f"{TUNED['gamma']} (tuned)", 'full')]),
    ('DQN exploration fraction', 'dqn',
     [(v + (' (tuned)' if v == TUNED['explore'] else ''), 'full' if v == TUNED['explore'] else f'explore{v}')
      for v in ['0.1', '0.3', '0.5']]),
    ('DQN learning rate', 'dqn', [('0.3x', 'lr_x0.3'), ('1x (tuned)', 'full'), ('3x', 'lr_x3')]),
    ('SAC learning rate', 'sac', [('0.3x', 'lr_x0.3'), ('1x (tuned)', 'full'), ('3x', 'lr_x3')]),
    ('SAC training length', 'sac', [('50 episodes (tuned)', 'full'), ('100 episodes', 'ep100')]),
    ('PPO entropy coefficient', 'ppo',
     [('0.01', 'ent0.01'), ('0.1', 'ent0.1'), (f"{TUNED['ent']} (tuned)", 'full')]),
]


def row(label, algo, run_name):
    path = os.path.join(BASE, algo, 'run', run_name)
    train, final = seed_rewards(path) if os.path.isdir(path) else ([], [])
    if not train:
        return f"  {label:22}{'-':>16}{'-':>16}"
    seeds = '' if len(train) == 5 else f"  ({len(train)}/5)"
    return (f"  {label:22}{statistics.mean(train):>9.0f} ± {statistics.pstdev(train):<4.0f}"
            f"{statistics.mean(final):>9.0f} ± {statistics.pstdev(final):<4.0f}{seeds}")


print(f"{'':24}{'train (last 5)':>16}{'final eval':>16}")
for title, algo, rows in EXPERIMENTS:
    print(title)
    for label, run_name in rows:
        print(row(label, algo, run_name))
    print()
