#!/bin/bash
#SBATCH --job-name=ce_train
#SBATCH -D .
#SBATCH -A bsc14
#SBATCH --qos=acc_debug
#SBATCH --output=logs_inference/jina_cross_%j.out
#SBATCH --error=logs_inference/jina_cross_%j.err
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

# /gpfs/projects/bsc14/scratch/.conda/factcheck/bin/python scripts/embeddings_jinav3/train.py --task_name monolingual\
#  --model_name /gpfs/projects/bsc14/abecerr1/hub/models--jinaai--jina-embeddings-v3/snapshots/fa78e35d523dcda8d3b5212c7487cf70a4b277da\
#   --output_path official/embeddings_jinav3 --task_file data/splits/tasks.json

/gpfs/projects/bsc14/scratch/.conda/factcheck/bin/python scripts/embeddings_jinav3/train.py --task_name crosslingual \
 --model_name /gpfs/projects/bsc14/abecerr1/hub/models--jinaai--jina-embeddings-v3/snapshots/fa78e35d523dcda8d3b5212c7487cf70a4b277da\
  --output_path output/embeddings_jinav3 --task_file data/splits/tasks.json --model_type 'jina'

# parser.add_argument('--task_name', type=str, required=True, help="Choose 'monolingual' or 'crosslingual'")
# parser.add_argument('--model_name', type=str, required=True, help="Path to the model")

# parser.add_argument('--output_path', type=str, default=None, help="Directory to save output")
# parser.add_argument('--task_file', type=str, default=None, help="Path to the task file")
# parser.add_argument('--langs', type=str, nargs='+', default=config.LANGS, help="List of languages")

