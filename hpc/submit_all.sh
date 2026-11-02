#!/bin/bash
# Submits the full experiment for one task: 5 seeds per algorithm, plus a stats job
# that starts once all of them have finished. PPO runs on CPU, DQN, SAC and DDPG on GPU.
# Usage: bash hpc/submit_all.sh <delf|abea> [algorithms...]
#   bash hpc/submit_all.sh abea          # all four algorithms
#   bash hpc/submit_all.sh delf ddpg     # only DDPG
# Set RUN_NAME to save results somewhere other than run/full, e.g.
#   RUN_NAME=noise0.3 NOISE_SIGMA=0.3 bash hpc/submit_all.sh delf ddpg
set -e
cd "$(dirname "$0")/.."

TASK=$1
shift
if [ "$TASK" != delf ] && [ "$TASK" != abea ]; then
    echo "Usage: bash hpc/submit_all.sh <delf|abea> [algorithms...]"
    exit 1
fi
ALGOS=("$@")
if [ ${#ALGOS[@]} -eq 0 ]; then
    ALGOS=(dqn ppo sac ddpg)
fi

GPU="--partition=course_gpu --gres=gpu:a100_10g:1 --cpus-per-task=1"

ids=""
for algo in "${ALGOS[@]}"; do
    case $algo in
        ppo)  opts="--time=02:00:00" ;;
        ddpg) opts="--time=06:00:00 $GPU" ;;
        dqn)  opts="--time=05:00:00 $GPU" ;;
        sac)  opts="--time=06:00:00 $GPU" ;;
        *) echo "Unknown algorithm: $algo"; exit 1 ;;
    esac
    id=$(sbatch --parsable --job-name=${TASK}_${algo}${RUN_NAME:+_$RUN_NAME} $opts hpc/train.sbatch $TASK $algo)
    echo "Submitted $TASK $algo ${RUN_NAME:-full}: $id"
    ids="$ids:$id"
done

stats=$(sbatch --parsable --dependency=afterok$ids hpc/stats.sbatch $TASK "${ALGOS[@]}")
echo "Submitted stats: $stats (starts when all runs finish)"
echo "Check progress with: bash hpc/status.sh"
