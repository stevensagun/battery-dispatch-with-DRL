#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --partition=course
#SBATCH --account=2026-fall-cs-669-tn294-ab3367
#SBATCH --qos=course
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=hpc/out/%x.%A_%a.out

# Full experiment for one task and algorithm: 5 seeds, one array task per seed (50 episodes each).
# Task is delf (DE-LF/REU) or abea (AB-EA). Results go to log/<env_name>/<algo>/run/full/run_<seed>.
# Usually submitted by hpc/submit_all.sh. By hand, from the repo root:
#   sbatch --job-name=delf_ppo hpc/train.sbatch delf ppo
#   sbatch --job-name=abea_dqn --partition=course_gpu --gres=gpu:a100_10g:1 --cpus-per-task=1 hpc/train.sbatch abea dqn
set -eo pipefail

TASK=$1
ALGO=$2
export RUN=$SLURM_ARRAY_TASK_ID

module purge > /dev/null 2>&1
module load wulver
module load Miniforge3
conda activate /course/2026/fall/cs/669/tn294/ab3367/envs/rl39

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "== $TASK $ALGO run_$RUN | node $(hostname) | partition $SLURM_JOB_PARTITION"
# Results folder under log/<env_name>/<algo>/run/ (default: full)
echo "${RUN_NAME:-full}" | python -m main.run_${ALGO}_${TASK}
