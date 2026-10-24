import os
import sys
import csv
import json
import statistics


# Summarize finished runs the same way as the paper's Tables 3 and 4: mean ± std across seeds of the
# average reward over the last 5 training episodes, in thousand C$. Only needs the Python standard library.
# Usage: python -m main.summarize_results <delf|abea> [results folder, default: full]
ENV_NAMES = {'delf': 'de1_bat_hes', 'abea': 'al4_bat_ea'}


def seed_rewards(path):
    """Per finished seed: average reward over the last 5 training episodes and final evaluation reward (k C$)."""
    with open(os.path.join(path, 'inputs.json')) as f:
        n_episodes = json.load(f)['EXP_PARAMS']['n_episodes']

    train, final = [], []
    for run in sorted(d for d in os.listdir(path) if d.startswith('run_')):
        output = os.path.join(path, run, 'output.json')
        if not os.path.exists(output):  # run not finished
            continue
        with open(os.path.join(path, run, 'train_monitor.csv')) as f:
            next(f)  # Monitor header line
            rewards = [float(row['r']) for row in csv.DictReader(f)]
        # The final evaluation episode is appended after the training episodes
        train.append(sum(rewards[n_episodes - 5:n_episodes]) / 5 / 1e3)
        with open(output) as f:
            final.append(json.load(f)['reward_sum'] / 1e3)
    return train, final


if __name__ == '__main__':
    base = os.path.join('log', ENV_NAMES[sys.argv[1]])
    run_name = sys.argv[2] if len(sys.argv) > 2 else 'full'
    print(f"{'model':6} {'seeds':>5} {'train (last 5 eps)':>20} {'final eval':>16}")
    for algo in ['dqn', 'ppo', 'sac', 'ddpg']:
        path = os.path.join(base, algo, 'run', run_name)
        if not os.path.isdir(path):
            continue
        train, final = seed_rewards(path)
        if train:
            print(f"{algo.upper():6} {len(train):>5} {statistics.mean(train):>12.0f} ± {statistics.pstdev(train):5.1f}"
                  f" {statistics.mean(final):>8.0f} ± {statistics.pstdev(final):5.1f}")
