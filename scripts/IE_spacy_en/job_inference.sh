#!/bin/bash
#SBATCH --job-name=ce_train
#SBATCH -D .
#SBATCH -A bsc14
#SBATCH --qos=acc_bscls
#SBATCH --output=logs_inference/ce_train_%j.out
#SBATCH --error=logs_inference/ce_train_%j.err
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

# ~/.conda/envs/factcheck/bin/python scripts/IE_spacy_en/train.py --task_file data/splits/tasks.json --task_name monolingual --model_name en_core_web_sm --output_path output/official/IE_spacy_en
~/.conda/envs/factcheck/bin/python scripts/IE_spacy_en/train.py --task_file data/splits/tasks.json  --task_name crosslingual --model_name en_core_web_sm --output_path output/IE_spacy_en
