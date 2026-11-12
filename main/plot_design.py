import os
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from main.summarize_results import ENV_NAMES, seed_rewards


# Plots the design-choice results (hpc/submit_design.sh) next to the paper's values: average reward over
# the last 5 training episodes, mean ± std across 5 seeds, in thousand C$.
# Usage: python -m main.plot_design   (writes figures/design_time_stack.png and figures/design_action_space.png)

TASKS = {'abea': 'AB-EA', 'delf': 'DE-LF/REU'}

# Paper, Table 4 (basic, time regular, time sin/cos, 2 obs., 4 obs.): (mean, std)
PAPER_T4 = {
    'abea': {'dqn': [(341, 6), (446, 6), (470, 8), (328, 13), (386, 33)],
             'ppo': [(339, 3), (348, 7), (380, 8), (330, 7), (330, 7)],
             'sac': [(331, 3), (412, 8), (423, 7), (334, 6), (403, 7)]},
    'delf': {'dqn': [(-2182, 43), (-1875, 12), (-1862, 9), (-1916, 24), (-1865, 9)],
             'ppo': [(-2245, 76), (-2210, 40), (-2166, 31), (-2184, 31), (-2156, 49)],
             'sac': [(-2333, 50), (-2264, 99), (-2145, 14), (-2128, 48), (-2198, 28)]},
}
T4_RUNS = ['basic', 'time_regular', 'time_sincos', 'stack2', 'stack4']
T4_LABELS = ['Basic', 'Time\nregular', 'Time\nsin/cos', '2 obs.\nstacked', '4 obs.\nstacked']

# Paper, Table 3 (continuous, 3 actions, 17 actions); DQN has no continuous version
PAPER_T3 = {
    'abea': {'ppo': [(339, 3.4), (332, 4.5), (330, 3.2)], 'dqn': [None, (341, 5.9), (334, 8.2)]},
    'delf': {'ppo': [(-2245, 75.9), (-2350, 31.8), (-2294, 60.1)], 'dqn': [None, (-2286, 68.2), (-2182, 42.7)]},
}
T3_LABELS = ['Continuous', '3 actions', '17 actions']
DQN_BASIC = {'abea': 1, 'delf': 2}  # which discretization DQN's basic run uses (3 or 17 actions)

NO_BATTERY = {'delf': -2364}  # paper: reward of a policy that never uses the battery on DE-LF/REU

OURS, PAPER = '#2a78d6', '#eb6834'
INK, INK_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'


def ours(task, algo, run_name):
    train, _ = seed_rewards(os.path.join('log', ENV_NAMES[task], algo, 'run', run_name))
    return (statistics.mean(train), statistics.pstdev(train)) if train else None


def t3_runs(task, algo):
    if algo == 'ppo':
        return ['basic', 'disc3', 'disc17']
    runs = [None, 'disc3', 'disc17']
    runs[DQN_BASIC[task]] = 'basic'
    return runs


def dots(ax, x, values, color, label):
    xs = [xi for xi, v in zip(x, values) if v]
    ms = [v[0] for v in values if v]
    ss = [v[1] for v in values if v]
    ax.errorbar(xs, ms, yerr=ss, fmt='o', color=color, ecolor=color, elinewidth=2, capsize=0,
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)


def style(ax, labels):
    ax.set_facecolor(SURFACE)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, color=INK_2)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', colors=INK_2, labelsize=9, length=0)
    ax.tick_params(axis='x', length=0)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)


def no_battery_line(ax, task, label):
    if task in NO_BATTERY:
        ax.axhline(NO_BATTERY[task], color=INK_2, linewidth=1, zorder=1)
        if label:
            ax.annotate('never uses the battery', (0, NO_BATTERY[task]), xycoords=('axes fraction', 'data'),
                        xytext=(4, -4), textcoords='offset points', ha='left', va='top', fontsize=8, color=INK_2)


def figure(columns, labels, paper, runs, title, path):
    fig, axes = plt.subplots(2, len(columns), figsize=(4.2 * len(columns), 7.2), sharey='row', facecolor=SURFACE)
    offset = 0.12
    for r, task in enumerate(TASKS):
        for c, algo in enumerate(columns):
            ax = axes[r, c]
            x = range(len(labels))
            names = runs(task, algo)
            dots(ax, [xi - offset for xi in x], [ours(task, algo, n) if n else None for n in names], OURS, 'Ours')
            dots(ax, [xi + offset for xi in x], paper[task][algo], PAPER, 'Paper')
            no_battery_line(ax, task, label=(algo == 'dqn'))
            style(ax, labels)
            ax.set_title(f'{algo.upper()}', fontsize=11, color=INK, loc='left')
            if c == 0:
                ax.set_ylabel(f'{TASKS[task]}\nreward (thousand C$)', fontsize=10, color=INK)
    handles = [Line2D([], [], marker='o', linestyle='', markersize=8, color=color) for color in (OURS, PAPER)]
    fig.legend(handles, ['Ours', 'Paper'], loc='upper right', ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.99, 0.985))
    fig.text(0.01, 0.985, title, ha='left', va='top', fontsize=13, color=INK)
    fig.text(0.01, 0.94, 'Average reward over the last 5 training episodes, mean ± std across 5 seeds',
             ha='left', va='top', fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print('Saved', path)


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    figure(['dqn', 'ppo', 'sac'], T4_LABELS, PAPER_T4, lambda task, algo: T4_RUNS,
           'Time counters and observation stacking (paper Table 4)', 'figures/design_time_stack.png')
    figure(['ppo', 'dqn'], T3_LABELS, PAPER_T3, t3_runs,
           'Action space, basic set-up (paper Table 3)', 'figures/design_action_space.png')
