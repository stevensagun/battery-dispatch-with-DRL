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

The `hpc/` folder has everything needed to run the DE-LF experiments (DQN, PPO and SAC, 5 seeds x 50 episodes each) on Wulver. All 15 runs train at the same time: PPO on CPU, DQN and SAC on A100 GPU slices. The whole experiment takes about 2-3 hours.

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
sbatch hpc/smoke.sbatch
```

### 5. Start the experiment

```bash
bash hpc/submit_all.sh
```

### 6. Check progress

```bash
bash hpc/status.sh
```

Jobs keep running after you log out. Slurm logs are in `hpc/out/`. To stop a job, use `scancel <jobid>`.

Never run training directly on the login node; always submit it with `sbatch` (the scripts above do this for you).

### 7. Results

Each run is saved in `log/de1_bat_hes/<algo>/run/full/run_<seed>/`. When all 15 runs are done, a stats job writes `stats.csv` and `eval_stats.csv` for each algorithm.

To copy the results to your laptop, run this on your laptop:

```bash
rsync -avz <your-ucid>@wulver.njit.edu:/course/2026/fall/cs/669/tn294/<your-ucid>/battery-dispatch-with-DRL/log/ ./log/
```
