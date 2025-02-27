#!/bin/bash
#SBATCH --job-name=ce_train
#SBATCH -D .
#SBATCH -A bsc14
#SBATCH --qos=acc_debug
#SBATCH --output=logs_ce_train/ce_train_%j.out
#SBATCH --error=logs_ce_train/ce_train_%j.err
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

$CONDA_PREFIX/bin/python scripts/contrastive/train.py\
    --task_name crosslingual\
    --teacher_model_name 'Snowflake/snowflake-arctic-embed-l-v2.0'\
    --reranker_model_name 'jinaai/jina-reranker-v2-base-multilingual'\
    --output_path tmp\
    --task_file data/splits/tasks.json