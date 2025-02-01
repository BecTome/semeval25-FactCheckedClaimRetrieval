#!/bin/bash


/home/zeus/miniconda3/envs/cloudspace/bin/python /teamspace/studios/this_studio/semeval25-FactCheckedClaimRetrieval/scripts/embeddings_xlm-r-multilingual-v1/train.py --task_name monolingual \
 --model_name sentence-transformers/paraphrase-xlm-r-multilingual-v1 \
  --output_path scripts/embeddings_xlm-r-multilingual-v1 --task_file data/splits/tasks_no_gs_overlap.json

# parser.add_argument('--task_name', type=str, required=True, help="Choose 'monolingual' or 'crosslingual'")
# parser.add_argument('--model_name', type=str, required=True, help="Path to the model")

# parser.add_argument('--output_path', type=str, default=None, help="Directory to save output")
# parser.add_argument('--task_file', type=str, default=None, help="Path to the task file")
# parser.add_argument('--langs', type=str, nargs='+', default=config.LANGS, help="List of languages")

