import os
import sys
import statistics

from main.summarize_results import ENV_NAMES, seed_rewards


# Prints the design-choice results (hpc/submit_design.sh) laid out like the paper's Table 4, plus the action
# spaces of Table 3: mean ± std across seeds of the average reward over the last 5 training episodes, in
# thousand C$. A cell shows (n/5) when only n seeds have finished.
# Usage: python -m main.summarize_design <delf|abea>
TASK = sys.argv[1]
BASE = os.path.join('log', ENV_NAMES[TASK])
DQN_DEFAULT = {'abea': 'disc3', 'delf': 'disc17'}[TASK]  # DQN's basic run uses this discretization


def cell(algo, run_name):
    path = os.path.join(BASE, algo, 'run', run_name)
    if not os.path.isdir(path):
        return '-'
    train, _ = seed_rewards(path)
    if not train:
        return '-'
    text = f"{statistics.mean(train):.0f} ± {statistics.pstdev(train):.0f}"
    return text if len(train) == 5 else f"{text} ({len(train)}/5)"


def table(title, columns, rows):
    print(title)
    print(f"{'':6}" + ''.join(f"{name:>18}" for name, _ in columns))
    for algo in rows:
        print(f"{algo.upper():6}" + ''.join(f"{cell(algo, run(algo)):>18}" for _, run in columns))
    print()


table('Time counters and observation stacking',
      [('basic', lambda a: 'basic'), ('time regular', lambda a: 'time_regular'),
       ('time sin/cos', lambda a: 'time_sincos'), ('2 obs stacked', lambda a: 'stack2'),
       ('4 obs stacked', lambda a: 'stack4')],
      ['dqn', 'ppo', 'sac'])
table('Action space (basic set-up)',
      [('continuous', lambda a: 'basic' if a == 'ppo' else '-'),
       ('3 actions', lambda a: 'basic' if a == 'dqn' and DQN_DEFAULT == 'disc3' else 'disc3'),
       ('17 actions', lambda a: 'basic' if a == 'dqn' and DQN_DEFAULT == 'disc17' else 'disc17')],
      ['ppo', 'dqn'])
