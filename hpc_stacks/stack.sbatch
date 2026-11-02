#!/bin/bash -l
#SBATCH --job-name=stack
#SBATCH --partition=course_gpu
#SBATCH --account=2026-fall-cs-669-tn294-srs73
#SBATCH --qos=course
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=hpc_stacks/out/%x.%j.out

# One Python experiment per job; its five seeds run sequentially.
# Usage from the repo root:
#   sbatch hpc_stacks/stack.sbatch run_transformer_abea_12Stack
set -eo pipefail

EXPERIMENT=${1:?Pass an experiment name without main. or .py}
case "$EXPERIMENT" in
    run_dqn_abea|run_dqn_abea_12Stack|run_dqn_abea_24Stack) ALGO=dqn ;;
    run_transformer_abea|run_transformer_abea_12Stack|run_transformer_abea_24Stack) ALGO=transformer ;;
    *) echo "Unknown stack experiment: $EXPERIMENT" >&2; exit 2 ;;
esac

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f "main/$EXPERIMENT.py"

module purge > /dev/null 2>&1
module load wulver
module load Miniforge3
conda activate /course/2026/fall/cs/669/tn294/srs73/envs/rl39

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLCONFIGDIR="${TMPDIR:-/tmp}/battery-matplotlib-$SLURM_JOB_ID"

# Job IDs keep repeated submissions from overwriting earlier results.
SAVE_FOLDER="${EXPERIMENT}_job${SLURM_JOB_ID}"
RESULT_PATH="log/al4_bat_ea/$ALGO/run/$SAVE_FOLDER"
mkdir -p "$RESULT_PATH"

echo "Experiment: main.$EXPERIMENT"
echo "Job: $SLURM_JOB_ID | node: $(hostname) | partition: $SLURM_JOB_PARTITION"
echo "Results: $RESULT_PATH"
echo "Training: five seeds sequentially; settings come from the Python script."
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
printf '%s\n' "$SAVE_FOLDER" | python -u -m "main.$EXPERIMENT"
echo "Completed: $RESULT_PATH (including stats.csv and eval_stats.csv)"
