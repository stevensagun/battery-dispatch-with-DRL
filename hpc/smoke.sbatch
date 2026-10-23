#!/bin/bash -l
#SBATCH --job-name=smoke
#SBATCH --partition=course
#SBATCH --account=2026-fall-cs-669-tn294-ab3367
#SBATCH --qos=course
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=hpc/out/%x.%j.out

# Short check: 1 seed x 1 episode per algorithm, logged to log/de1_bat_hes/<algo>/run/smoke_<partition>.
# Submit from the repo root:
#   CPU: sbatch hpc/smoke.sbatch
#   GPU: sbatch --partition=course_gpu --gres=gpu:a100_10g:1 --job-name=smoke_gpu hpc/smoke.sbatch dqn sac
set -eo pipefail

module purge > /dev/null 2>&1
module load wulver
module load Miniforge3
conda activate /course/2026/fall/cs/669/tn294/ab3367/envs/rl39

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export N_RUNS=1 N_EPISODES=1

ALGOS=("$@")
if [ ${#ALGOS[@]} -eq 0 ]; then
    ALGOS=(dqn ppo sac)
fi

echo "node $(hostname) | partition $SLURM_JOB_PARTITION | cpus $SLURM_CPUS_PER_TASK"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

for algo in "${ALGOS[@]}"; do
    echo "== $algo"
    echo "smoke_$SLURM_JOB_PARTITION" | python -m main.run_${algo}_delf
done
