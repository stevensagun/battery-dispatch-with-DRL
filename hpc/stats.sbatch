#!/bin/bash -l
#SBATCH --job-name=stats
#SBATCH --partition=course
#SBATCH --account=2026-fall-cs-669-tn294-ab3367
#SBATCH --qos=course
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=hpc/out/%x.%j.out

# Builds stats.csv / eval_stats.csv for each algorithm once all its seeds have finished.
# Submit from the repo root after the delf jobs:
#   sbatch --dependency=afterok:<dqn_id>:<ppo_id>:<sac_id> hpc/stats.sbatch dqn ppo sac
set -eo pipefail

module purge > /dev/null 2>&1
module load wulver
module load Miniforge3
conda activate /course/2026/fall/cs/669/tn294/ab3367/envs/rl39

for algo in "$@"; do
    python -m main.make_stats log/de1_bat_hes/$algo/run/full
done
