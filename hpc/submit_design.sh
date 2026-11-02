#!/bin/bash
# Submits the design-choice experiments (paper Tables 3 and 4) for one task, 5 seeds each.
# Each variant changes one thing compared to the basic set-up (see utils/variants.py):
#   - time counters (regular, sine/cosine) and observation stacking (2, 4) for DQN, PPO and SAC
#   - action space: PPO with 3 and 17 discrete actions, DQN with the discretization it doesn't use by default
# Results go to log/<env_name>/<algo>/run/<variant>.
# Usage: bash hpc/submit_design.sh <delf|abea>
set -e
cd "$(dirname "$0")/.."

TASK=$1
case $TASK in
    abea) DQN_OTHER=disc17 ;;  # DQN uses 3 actions on AB-EA by default
    delf) DQN_OTHER=disc3 ;;   # and 17 on DE-LF
    *) echo "Usage: bash hpc/submit_design.sh <delf|abea>"; exit 1 ;;
esac

for v in basic time_regular time_sincos stack2 stack4; do
    RUN_NAME=$v VARIANT=$v bash hpc/submit_all.sh $TASK dqn ppo sac
done
for v in disc3 disc17; do
    RUN_NAME=$v VARIANT=$v bash hpc/submit_all.sh $TASK ppo
done
RUN_NAME=$DQN_OTHER VARIANT=$DQN_OTHER bash hpc/submit_all.sh $TASK dqn
