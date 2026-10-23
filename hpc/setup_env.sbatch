#!/bin/bash -l
#SBATCH --job-name=setup_env
#SBATCH --partition=course
#SBATCH --account=2026-fall-cs-669-tn294-ab3367
#SBATCH --qos=course
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=hpc/out/%x.%j.out

# Builds the Python 3.9 env (gym 0.21 / stable-baselines3 1.7) in the course directory.
# Submit from the repo root: sbatch hpc/setup_env.sbatch
set -eo pipefail

COURSE_DIR=/course/2026/fall/cs/669/tn294/ab3367
ENV_DIR=$COURSE_DIR/envs/rl39
export CONDA_PKGS_DIRS=$COURSE_DIR/.conda_pkgs

module purge > /dev/null 2>&1
module load wulver
module load Miniforge3

if [ ! -d "$ENV_DIR" ]; then
    conda create -y -p "$ENV_DIR" python=3.9
fi
conda activate "$ENV_DIR"

# gym 0.21 has a setup.py that newer setuptools/wheel/pip reject
python -m pip install --no-cache-dir "setuptools==65.5.0" "wheel<0.40" "pip<24"
python -m pip install --no-cache-dir --no-build-isolation gym==0.21.0
python -m pip install --no-cache-dir stable-baselines3==1.7.0 torch==1.13.1 pandas tqdm tensorboard "numpy<1.24"

python -c "import torch, stable_baselines3, gym; print('sb3', stable_baselines3.__version__, '| gym', gym.__version__, '| torch', torch.__version__)"
