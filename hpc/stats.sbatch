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
# Usually submitted by hpc/submit_all.sh. By hand, from the repo root:
#   sbatch --dependency=afterok:<job ids> hpc/stats.sbatch delf dqn ppo sac ddpg
set -eo pipefail

TASK=$1
shift
case $TASK in
    delf) ENV_NAME=de1_bat_hes ;;
    abea) ENV_NAME=al4_bat_ea ;;
    *) echo "Unknown task: $TASK (use delf or abea)"; exit 1 ;;
esac

module purge > /dev/null 2>&1
module load wulver
module load Miniforge3
conda activate /course/2026/fall/cs/669/tn294/ab3367/envs/rl39

for algo in "$@"; do
    python -m main.make_stats log/$ENV_NAME/$algo/run/${RUN_NAME:-full}
done
