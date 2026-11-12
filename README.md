# Battery Dispatch with Deep Reinforcement Learning

This repo contains the code to the paper:

_Deep Reinforcement Learning for Economic Battery Dispatch: A Comprehensive Comparison of Algorithms and Experiment Design Choices_
by Manuel Sage & Yaoyao Fiona Zhao.

Published at the Journal of Energy Storage: https://doi.org/10.1016/j.est.2025.115428

### Note:

- The data used in the case studies can be found in the /data folder and the parameters in /environments/env_params.py.
- The discharge behavior is handled slightly different in the code than in Eq. (1) in the paper.
- Since uploading this repo, we have updated and improved our codebase for additional research projects:
  - Battery arbitrage with time-series forecasting and RL:
    - Paper: https://doi.org/10.1115/ES2024-130538
    - Preprint: https://arxiv.org/abs/2410.20005
    - Repository: https://github.com/masa2203/battery_arbitrage_with_drl
  - Joint dispatch of gas turbines and batteries with RL:
    - Paper: https://doi.org/10.1016/j.egyai.2025.100653
    - Preprint: https://dx.doi.org/10.2139/ssrn.5117155
    - Repository: https://github.com/masa2203/joint_bes_gt_dispatch

# Prerequisites

## Create environment

Have the following installed:

- Conda

Run commands:

```bash
conda env create -f environment-linux.yml
conda activate energy-arbitrage
```

# How to run

From the root:

```bash
python -m main.[name]
```

where [name] is one of the files. For example:

```bash
python -m main.run_dqn
```

# Running on Wulver (NJIT HPC)

The `hpc/` folder has everything needed to run both case studies on Wulver with all four algorithms (DQN, PPO, SAC and DDPG, 5 seeds x 50 episodes each):

- `delf`: DE-LF/REU (Germany, load following)
- `abea`: AB-EA (Alberta, energy arbitrage)

All runs for a task train at the same time: PPO on CPU, DQN, SAC and DDPG on A100 GPU slices. A full task takes about 2-3 hours.

The paper doesn't report tuned DDPG hyperparameters (it drops DDPG after the basic experiments). The original settings in `run_ddpg.py` have no exploration noise and only 5 gradient updates per episode, so `run_ddpg_delf.py` and `run_ddpg_abea.py` instead reuse the tuned SAC setup for each task and add Gaussian exploration noise.

### 1. Log in

Connect to the NJIT VPN if you're off campus, then:

```bash
ssh <your-ucid>@wulver.njit.edu
```

### 2. Get the code

Course work has to live in your course directory:

```bash
cd /course/2026/fall/cs/669/tn294/<your-ucid>
git clone https://github.com/stevensagun/battery-dispatch-with-DRL.git
cd battery-dispatch-with-DRL
```

The job scripts are set up for UCID `ab3367`. If that's not you, swap in yours:

```bash
sed -i "s/ab3367/$USER/g" hpc/*.sbatch
```

### 3. Build the Python environment (only once)

```bash
sbatch hpc/setup_env.sbatch
```

This takes about 6 minutes. It installs Python 3.9 with gym 0.21, stable-baselines3 1.7 and torch 1.13 into `envs/rl39` in your course directory.

### 4. Quick test (optional)

Runs each algorithm for one episode to make sure everything works:

```bash
sbatch hpc/smoke.sbatch delf
```

### 5. Start the experiment

Pick the task (`delf` or `abea`):

```bash
bash hpc/submit_all.sh delf
```

This runs all four algorithms. To run only some of them, list them after the task:

```bash
bash hpc/submit_all.sh delf ddpg
```

### 6. Check progress

```bash
bash hpc/status.sh
```

Jobs keep running after you log out. Slurm logs are in `hpc/out/`. To stop a job, use `scancel <jobid>`.

Never run training directly on the login node; always submit it with `sbatch` (the scripts above do this for you).

### 7. Results

Each run is saved in `log/<case study>/<algo>/run/full/run_<seed>/`, where the case study folder is `de1_bat_hes` (DE-LF) or `al4_bat_ea` (AB-EA). When all runs for a task are done, a stats job writes `stats.csv` and `eval_stats.csv` for each algorithm.

To print a results table in the same format as the paper (average reward over the last 5 training episodes, mean ± std across seeds, in thousand C$):

```bash
python -m main.summarize_results delf
```

To copy the results to your laptop, run this on your laptop:

```bash
rsync -avz <your-ucid>@wulver.njit.edu:/course/2026/fall/cs/669/tn294/<your-ucid>/battery-dispatch-with-DRL/log/ ./log/
```

## Design-choice experiments

These repeat the paper's comparisons of time counters and observation stacking (Table 4) and action spaces (Table 3) for DQN, PPO and SAC. Each experiment starts from the basic set-up (no time counters, no stacking, no reward penalties) and changes one thing:

| Variant | What changes |
|---|---|
| `basic` | nothing |
| `time_regular` | adds hour, week and month counters |
| `time_sincos` | adds sine/cosine-encoded hour, week and month |
| `stack2`, `stack4` | stacks the last 2 or 4 observations |
| `disc3`, `disc17` | 3 or 17 discrete actions (PPO and DQN) |

To run all of them for one task (5 seeds each, about 90 runs):

```bash
bash hpc/submit_design.sh delf
```

To print the results laid out like the paper's tables:

```bash
python -m main.summarize_design delf
```

To plot both tasks next to the paper's values (saved in `figures/`):

```bash
python -m main.plot_design
```

To run a single variant yourself, set `VARIANT` (and `RUN_NAME` for the results folder):

```bash
RUN_NAME=stack4 VARIANT=stack4 bash hpc/submit_all.sh abea sac
```

## Hyperparameter experiments

These are our own experiments (not in the paper). Each run keeps the tuned set-up from Appendix C and changes one hyperparameter:

| Algorithm | Hyperparameter | Values |
|---|---|---|
| DQN | discount factor | 0.9, 0.95, 0.99, 0.999 |
| DQN | exploration (fraction of training over which epsilon decays) | 0.1, 0.3, 0.5 |
| DQN, SAC | learning rate | 0.3x, 1x, 3x the tuned value |
| PPO | entropy coefficient | 0.01, 0.1, tuned |
| SAC | training length | 50, 100 episodes |

Values equal to the tuned one aren't run again; the summary takes them from `run/full`. To run all of them for one task (5 seeds each, 65 runs):

```bash
bash hpc/submit_hparams.sh delf
```

To print the results:

```bash
python -m main.summarize_hparams delf
```

To run a single setting yourself, set `VARIANT` to the name and value, for example `gamma0.99`, `explore0.3`, `lr_x3` or `ent0.1`:

```bash
RUN_NAME=gamma0.99 VARIANT=gamma0.99 bash hpc/submit_all.sh abea dqn
```
