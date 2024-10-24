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

tasks_path = "data/splits/tasks_no_gs_overlap.json"
posts_path = "data/complete_data/posts.csv"
fact_checks_path = "data/complete_data/fact_checks.csv"
gs_path = "data/complete_data/pairs.csv"

# if available
embedings_path = "scripts/all-MimiLM-L6-v2/monolingual_embeddings.json"
processed_data_path = "scripts/all-MimiLM-L6-v2/processedData"

langs = ['fra', 'spa', 'eng', 'por', 'tha', 'deu', 'msa', 'ara']
trans_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
# trans_model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
cross_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# check if cuda is available, if not use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

for lang in tqdm(langs, desc="Languages"):
    
    print(f"Processing language: {lang}")
    
    if not os.path.isdir(processed_data_path):
        posts = TextConcatPosts(posts_path, tasks_path, task_name="monolingual", gs_path=gs_path, lang=lang, demojize=True, prefix="query: ", version="english")
        fact_checks = TextConcatFactCheck(fact_checks_path, tasks_path, task_name="monolingual", lang=lang, demojize=True, prefix="passage: ", version="english")
        df_fc = fact_checks.df
        df_posts_dev = posts.df_dev
        print(f"Data processed! for language {lang}\n")
    else:
        df_fc = pd.read_csv(f"scripts/all-MimiLM-L6-v2/processedData/fact_checks_{lang}.csv")
        df_posts_train = pd.read_csv(f"scripts/all-MimiLM-L6-v2/processedData/posts_train_{lang}.csv")
        df_posts_dev = pd.read_csv(f"scripts/all-MimiLM-L6-v2/processedData/posts_dev_{lang}.csv")
        print(f"Data loaded! for language {lang}\n")

    model = EmbeddingModel(trans_model_name, df_fc, device=device, k=100)
    
    df_posts_dev["emb_preds"] = model.predict(df_posts_dev["full_text"].values).tolist()
        
    cross_model = CrossencoderModel(cross_model_name, df_posts_dev, show_progress_bar=False, batch_size=512, k=10, device=device)

    df_posts_dev["preds"] = df_posts_dev.apply(lambda x:cross_model.predict(x["full_text"], x["emb_preds"]), axis=1)

    print(cross_model.evaluate(df_posts_dev, task_name='monolingual', lang=lang))
        