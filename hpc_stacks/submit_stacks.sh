#!/bin/bash
# Default: submit DQN and transformer with 12- and 24-frame histories.
# Select specific experiments by passing their filenames without .py.
set -eo pipefail
cd "$(dirname "$0")/.."

EXPERIMENTS=("$@")
if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    EXPERIMENTS=(
        run_dqn_abea_12Stack
        run_dqn_abea_24Stack
        run_transformer_abea_12Stack
        run_transformer_abea_24Stack
    )
fi

# Validate every selection before submitting any jobs.
for experiment in "${EXPERIMENTS[@]}"; do
    case "$experiment" in
        run_dqn_abea|run_dqn_abea_12Stack|run_dqn_abea_24Stack|run_transformer_abea|run_transformer_abea_12Stack|run_transformer_abea_24Stack) ;;
        *) echo "Unknown stack experiment: $experiment" >&2; exit 2 ;;
    esac
    test -f "main/$experiment.py"
done

mkdir -p hpc_stacks/out
for experiment in "${EXPERIMENTS[@]}"; do
    job=$(sbatch --parsable --job-name="stack_${experiment#run_}" hpc_stacks/stack.sbatch "$experiment")
    echo "$experiment: job $job"
done
echo "Check progress: bash hpc_stacks/status_stacks.sh"
