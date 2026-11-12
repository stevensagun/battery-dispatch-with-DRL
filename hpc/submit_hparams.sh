#!/bin/bash
# Submits our own hyperparameter experiments (Milestone 2) for one task, 5 seeds each. Every run keeps the
# tuned set-up and changes one hyperparameter. Values equal to the tuned one are skipped: run/full has them.
#   DQN: discount factor 0.9, 0.95, 0.99, 0.999; exploration over 10%, 30%, 50% of training;
#        learning rate 0.3x and 3x
#   SAC: learning rate 0.3x and 3x; 100 instead of 50 episodes
#   PPO: entropy coefficient 0.01 and 0.1
# Usage: bash hpc/submit_hparams.sh <delf|abea>
set -e
cd "$(dirname "$0")/.."

TASK=$1
case $TASK in
    abea) EXPLORE="0.3 0.5" ;;  # tuned: 0.1
    delf) EXPLORE="0.1 0.3" ;;  # tuned: 0.5
    *) echo "Usage: bash hpc/submit_hparams.sh <delf|abea>"; exit 1 ;;
esac

for g in 0.9 0.95 0.99 0.999; do
    RUN_NAME=gamma$g VARIANT=gamma$g bash hpc/submit_all.sh $TASK dqn
done
for e in $EXPLORE; do
    RUN_NAME=explore$e VARIANT=explore$e bash hpc/submit_all.sh $TASK dqn
done
for x in 0.3 3; do
    RUN_NAME=lr_x$x VARIANT=lr_x$x bash hpc/submit_all.sh $TASK dqn sac
done
for c in 0.01 0.1; do
    RUN_NAME=ent$c VARIANT=ent$c bash hpc/submit_all.sh $TASK ppo
done
RUN_NAME=ep100 N_EPISODES=100 TIME=08:00:00 bash hpc/submit_all.sh $TASK sac
