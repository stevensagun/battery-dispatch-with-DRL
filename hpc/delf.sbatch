#!/bin/bash -l
#SBATCH --job-name=delf
#SBATCH --partition=course
#SBATCH --account=2026-fall-cs-669-tn294-ab3367
#SBATCH --qos=course
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=hpc/out/%x.%A_%a.out

# Full DE-LF experiment for one algorithm: 5 seeds, one array task per seed (50 episodes each).
# Results go to log/de1_bat_hes/<algo>/run/full/run_<seed>.
# Submit from the repo root, one job per algorithm:
#   sbatch --job-name=delf_ppo hpc/delf.sbatch ppo
#   sbatch --job-name=delf_dqn --partition=course_gpu --gres=gpu:a100_10g:1 hpc/delf.sbatch dqn
set -eo pipefail

ALGO=$1
export RUN=$SLURM_ARRAY_TASK_ID

module purge > /dev/null 2>&1
module load wulver
module load Miniforge3
conda activate /course/2026/fall/cs/669/tn294/ab3367/envs/rl39

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "== $ALGO run_$RUN | node $(hostname) | partition $SLURM_JOB_PARTITION"
echo full | python -m main.run_${ALGO}_delf
