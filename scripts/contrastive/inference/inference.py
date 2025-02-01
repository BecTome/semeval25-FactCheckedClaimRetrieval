import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
from time import time
import torch
import os
# os.chdir("../../..")
print(os.getcwd())
import sys
sys.path.append(".")

from src import config
from src.datasets import TextConcatFactCheck, TextConcatPosts
from src.models import EmbeddingModel

def predict_fact_checks(teacher_model_base, reranker_model_name, task_name="monolingual", n_candidates=100, output_k=10, lang="eng"):
    """
    Function to predict the top fact-check candidates for posts.

    Args:
    - model_path (str): Path to the pre-trained model.
    - task_name (str): Task name, default is "monolingual".
    - n_candidates (int): Number of candidate fact-checks to retrieve.
    - output_k (int): Number of top-ranked fact-checks to output.
    - lang (str): Language code for the task.

    Returns:
    - pd.DataFrame: DataFrame containing posts with predictions.
    """
    
    # Hardcoded paths
    fact_checks_path = "data/test_data/fact_checks.csv"
    posts_path = "data/test_data/posts.csv"
    tasks_path = "data/splits/splits_test.json"
    gs_path = config.GS_PATH

    # Load Fact Checks
    print("Loading Fact Checks...")
    fc = TextConcatFactCheck(fact_checks_path, tasks_path=tasks_path, task_name=task_name, lang=lang, version="english")

    # Load Posts
    print("Loading Posts...")
    posts = TextConcatPosts(posts_path, tasks_path=tasks_path, task_name=task_name, lang=lang, gs_path=gs_path, version="english")

    teacher_model = EmbeddingModel(teacher_model_base, fc.df, batch_size=emb_batch_size)

    # Initialize Models
    print("Initializing Models...")
    rerank_model = CrossEncoder(reranker_model_name, num_labels=1, max_length=1024, trust_remote_code=True)

    
    # Predict Fact-Check Candidates
    print("Predicting...")
    time_start = time()
    with torch.no_grad():
        arr_cands = teacher_model.predict(posts.df_dev["full_text"].values)[:,:n_candidates]

    # Initialize List for Predictions
    ls_preds = []
    for i in tqdm(range(len(posts.df_dev))):
        ranked_vals = rerank_model.rank(posts.df_dev["full_text"].values[i], fc.df.loc[arr_cands[i], "full_text"].values, show_progress_bar=False)
        ls_preds.append([int(arr_cands[i, dd["corpus_id"]]) for dd in ranked_vals][:output_k])

    # Add Predictions to DataFrame
    posts.df_dev["preds"] = ls_preds
    print(f"Time taken: {time() - time_start:.2f}s\n")

    d_out = posts.df_dev["preds"].to_dict()

    return d_out

import os
from src.utils import log_info

teacher_model_name = '/gpfs/projects/bsc14/abecerr1/hub/models--Snowflake--snowflake-arctic-embed-l-v2.0/snapshots/edc2df7b6c25794b340229ca082e7c78782e6374'
teacher_model_base = SentenceTransformer(teacher_model_name, device="cuda", trust_remote_code=True)
emb_batch_size = 64

d_monolingual = {}
d_crosslingual = {}

preds_path = "official/contrastive/snowflake_mv2/predictions"
models_path = "official/contrastive/snowflake_mv2/models"

# Save Predictions
import json

if not os.path.exists(preds_path):
    os.makedirs(preds_path)
    
log_info("Predicting Monolingual and Crosslingual Fact-Checks...")
for model_name in os.listdir(models_path):
    log_info(f"Predicting for model: {model_name}")
    task = model_name.split("_")[0]
    lang = model_name.split("_")[1]
    
    task = "monolingual" if lang in ["pol", "tur"] else task
    
    model_name = os.path.join(models_path, model_name)

    d_out = predict_fact_checks(teacher_model_base, model_name, task_name=task, n_candidates=100, output_k=10, lang=lang)
    
    if task == "monolingual":
        d_monolingual.update(d_out)
        log_info("Saving Predictions to " + preds_path)
        json.dump(d_monolingual, open(os.path.join(preds_path, "monolingual_predictions.json"), "w"))
    else:
        d_crosslingual.update(d_out)
        log_info("Monolingual Predictions saved.")
        json.dump(d_crosslingual, open(os.path.join(preds_path, "crosslingual_predictions.json"), "w"))
        log_info("Crosslingual Predictions saved.")

log_info("====================================")




