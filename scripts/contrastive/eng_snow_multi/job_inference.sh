#!/bin/bash
#SBATCH --job-name=ce_train
#SBATCH -D .
#SBATCH -A bsc14
#SBATCH --qos=acc_bscls
#SBATCH --output=output/contrastive/snowflake/logs/log_%j.out
#SBATCH --error=output/contrastive/snowflake/logs/log_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --exclusive

## --qos=acc_bscls
module load anaconda

# Initialize conda for bash shell
conda init bash
source ~/.bashrc  # This reloads the shell to apply conda settings

conda activate factcheck

/gpfs/projects/bsc14/scratch/.conda/factcheck/bin/python scripts/contrastive/eng_snow_multi/train.py --task_name crosslingual\
 --teacher_model_name '/gpfs/projects/bsc14/abecerr1/hub/models--Snowflake--snowflake-arctic-embed-l-v2.0/snapshots/edc2df7b6c25794b340229ca082e7c78782e6374'\
 --reranker_model_name '/gpfs/projects/bsc14/abecerr1/hub/models--jinaai--jina-reranker-v2-base-multilingual/snapshots/126747772a932960028d9f4dc93bd5d9c4869be4'\
  --output_path output/official/contrastive/snowflake --task_file data/splits/tasks.json

# /gpfs/projects/bsc14/scratch/.conda/factcheck/bin/python scripts/contrastive/train.py --task_name monolingual\
#  --teacher_model_name '/gpfs/projects/bsc14/abecerr1/hub/models--intfloat--multilingual-e5-large/snapshots/ab10c1a7f42e74530fe7ae5be82e6d4f11a719eb'\
#  --reranker_model_name 'jinaai/jina-reranker-v2-base-multilingual'\
#   --output_path official/contrastive --task_file data/splits/tasks.json

# /gpfs/projects/bsc14/scratch/.conda/factcheck/bin/python scripts/embeddings_jinav3/train.py --task_name crosslingual \
#  --model_name /gpfs/projects/bsc14/abecerr1/hub/models--jinaai--jina-embeddings-v3/snapshots/fa78e35d523dcda8d3b5212c7487cf70a4b277da\
#   --output_path output/embeddings_jinav3 --task_file data/splits/tasks.json

# parser.add_argument('--task_name', type=str, required=True, help="Choose 'monolingual' or 'crosslingual'")
# parser.add_argument('--model_name', type=str, required=True, help="Path to the model")

# parser.add_argument('--output_path', type=str, default=None, help="Directory to save output")
# parser.add_argument('--task_file', type=str, default=None, help="Path to the task file")
# parser.add_argument('--langs', type=str, nargs='+', default=config.LANGS, help="List of languages")