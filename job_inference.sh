#!/bin/bash
#SBATCH --job-name=ce_train
#SBATCH -D .
#SBATCH -A bsc14
#SBATCH --qos=acc_debug
#SBATCH --output=scripts/contrastive/inference/logs/log_%j.out
#SBATCH --error=scripts/contrastive/inference/logs/log_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:00

## --qos=acc_bscls
module load anaconda

# Initialize conda for bash shell
conda init bash
source ~/.bashrc  # This reloads the shell to apply conda settings

conda activate factcheck

python scripts/contrastive/inference/inference.py
