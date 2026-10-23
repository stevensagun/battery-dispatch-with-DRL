#!/bin/bash
# Submits the full DE-LF experiment: PPO on CPU, DQN and SAC on GPU (5 seeds each),
# plus a stats job that starts once all 15 runs have finished.
# Usage: bash hpc/submit_all.sh
set -e
cd "$(dirname "$0")/.."

GPU="--partition=course_gpu --gres=gpu:a100_10g:1 --cpus-per-task=1"

ppo=$(sbatch --parsable --job-name=delf_ppo --time=02:00:00 hpc/delf.sbatch ppo)
dqn=$(sbatch --parsable --job-name=delf_dqn --time=05:00:00 $GPU hpc/delf.sbatch dqn)
sac=$(sbatch --parsable --job-name=delf_sac --time=06:00:00 $GPU hpc/delf.sbatch sac)
stats=$(sbatch --parsable --dependency=afterok:$ppo:$dqn:$sac hpc/stats.sbatch dqn ppo sac)

echo "Submitted: ppo=$ppo dqn=$dqn sac=$sac stats=$stats"
echo "Check progress with: bash hpc/status.sh"
