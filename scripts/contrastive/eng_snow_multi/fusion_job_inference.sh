#!/bin/bash
#SBATCH --job-name=ce_train
#SBATCH -D .
#SBATCH -A bsc14
#SBATCH --qos=acc_bscls
#SBATCH --output=output/official/contrastive/fusion_snowflake/logs/log_%j.out
#SBATCH --error=output/official/contrastive/fusion_snowflake/logs/log_%j.err
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

# /gpfs/projects/bsc14/scratch/.conda/factcheck/bin/python scripts/contrastive/eng_snow_multi/train_fusion.py --task_name monolingual\
#  --teacher_model_name '/gpfs/projects/bsc14/abecerr1/hub/models--Snowflake--snowflake-arctic-embed-l-v2.0/snapshots/edc2df7b6c25794b340229ca082e7c78782e6374'\
#  --reranker_model_name '/gpfs/projects/bsc14/abecerr1/hub/models--jinaai--jina-reranker-v2-base-multilingual/snapshots/126747772a932960028d9f4dc93bd5d9c4869be4'\
#   --output_path output/contrastive/fusion_snowflake --task_file data/splits/tasks_no_gs_overlap.json

/gpfs/projects/bsc14/scratch/.conda/factcheck/bin/python scripts/contrastive/eng_snow_multi/train_fusion.py --task_name monolingual \
 --dense_model_name '/gpfs/projects/bsc14/abecerr1/hub/models--Snowflake--snowflake-arctic-embed-l-v2.0/snapshots/edc2df7b6c25794b340229ca082e7c78782e6374' \
 --sparse_model_name 'BM25-PT' \
 --reranker_model_name '/gpfs/projects/bsc14/abecerr1/hub/models--jinaai--jina-reranker-v2-base-multilingual/snapshots/126747772a932960028d9f4dc93bd5d9c4869be4' \
 --output_path output/official/contrastive/fusion_snowflake --task_file data/splits/tasks.json