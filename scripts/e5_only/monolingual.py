#!/usr/bin/env python

from encodings import normalize_encoding
from multiprocessing import process
from networkx import hits
from numpy import cross
import pandas as pd

from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch

from tqdm import tqdm
import json

import sys
import os

from zmq import device

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.datasets import TextConcatFactCheck, TextConcatPosts
from src.models import EmbeddingModel, CrossencoderModel
from src import config

tasks_path = config.TASKS_PATH
posts_path = config.POSTS_PATH
fact_checks_path = config.FACT_CHECKS_PATH
gs_path = config.GS_PATH
langs = config.LANGS
# 

# if available
embedings_path = "scripts/all-MimiLM-L6-v2/monolingual_embeddings.json"
processed_data_path = "scripts/all-MimiLM-L6-v2/processedData"

# models to use
trans_model_name = config.E5ENCODER
cross_model_name = config.ROBERTA_CROSS

output_path = os.path.join(config.OUTPUT_PATH, trans_model_name.split("/")[-1] + "_" + cross_model_name.split("/")[-1] + '.csv')

# check if cuda is available, if not use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
results = []
for lang in tqdm(langs, desc="Languages"):
    
    print(f"Processing language: {lang}")
    
    if not os.path.isdir(processed_data_path):
        posts = TextConcatPosts(posts_path, tasks_path, task_name="monolingual", gs_path=gs_path, lang=lang, demojize=True, prefix="query: ",version="original")
        fact_checks = TextConcatFactCheck(fact_checks_path, tasks_path, task_name="monolingual", lang=lang, demojize=True, prefix="passage: ",version="original")
        df_fc = fact_checks.df
        df_posts_dev = posts.df_dev
        print(f"Data processed! for language {lang}\n")
    else:
        df_fc = pd.read_csv(f"scripts/all-MimiLM-L6-v2/processedData/fact_checks_{lang}.csv")
        # df_posts_train = pd.read_csv(f"scripts/all-MimiLM-L6-v2/processedData/posts_train_{lang}.csv")
        df_posts_dev = pd.read_csv(f"scripts/all-MimiLM-L6-v2/processedData/posts_dev_{lang}.csv")
        print(f"Data loaded! for language {lang}\n")

    print(f"Creating Embedding Model for language: {lang}")
    model = EmbeddingModel(trans_model_name, df_fc, device=device, k=100)
    
    print(f"Predicting Embeddings for language: {lang}")
    df_posts_dev["preds"] = model.predict(df_posts_dev["full_text"].values).tolist()
    
    # print(f"Creating Crossencoder Model for language: {lang}")
    # cross_model = CrossencoderModel(cross_model_name, df_fc, show_progress_bar=False, batch_size=512, k=10, device=device)

    # print(f"Reranking for language: {lang}")
    # df_posts_dev["preds"] = df_posts_dev.apply(lambda x:cross_model.predict(x["full_text"], x["emb_preds"]), axis=1)

    results.append(model.evaluate(df_posts_dev, task_name='monolingual', lang=lang))

print(f"Results: {results}")

# Convert list of dictionaries into a single dictionary
flattened_data = {k: v for task in results for d in task for k, v in task[d].items()}

# Convert to DataFrame
df = pd.DataFrame(flattened_data).T

# Rename index and columns for clarity
df.index.name = 'language'
df.columns = ['1', '3', '5', '10']

# add average columns and row
df.loc['average'] = df.mean()
df.round(3)

# Display the DataFrame
print(df)
df.to_csv(output_path, index=True)


