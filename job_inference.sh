#!/bin/bash
#SBATCH --job-name=ce_train
#SBATCH -D .
#SBATCH -A bsc14
#SBATCH --qos=acc_debug
#SBATCH --output=example_log/ce_train_%j.out
#SBATCH --error=example_log/ce_train_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:00
#SBATCH --exclusive

## --qos=acc_bscls
module load anaconda

# Initialize conda for bash shell
conda init bash
source ~/.bashrc  # This reloads the shell to apply conda settings

conda activate factcheck

/gpfs/projects/bsc14/scratch/.conda/factcheck/bin/python scripts/embeddings_large_e5/train_predict.py
