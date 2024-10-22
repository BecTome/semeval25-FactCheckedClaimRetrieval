#!/usr/bin/env python

from encodings import normalize_encoding
from SimpleITK import namedtuple
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
from src.models import EmbeddingModel

tasks_path = "data/splits/tasks_no_gs_overlap.json"
posts_path = "data/complete_data/posts.csv"
fact_checks_path = "data/complete_data/fact_checks.csv"
gs_path = "data/complete_data/pairs.csv"
embedings_path = "scripts/all-MimiLM-L6-v2/monolingual_embeddings.json"
langs = ['fra', 'spa', 'eng', 'por', 'tha', 'deu', 'msa', 'ara']
trans_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
cross_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

device = "cpu"

#check if the embeddings are already computed
embedings = None
if os.path.exists(embedings_path):
    print("Embeddings already computed... Loading them!")
    with open(embedings_path, "r") as f:
        embedings = json.load(f)
    print("Embeddings loaded!\n")
    

d_out = {}
emb_out = {}
for lang in tqdm(langs, desc="Languages"):

    posts = TextConcatPosts(posts_path, tasks_path, task_name="monolingual", gs_path=gs_path, lang=lang)
    fact_checks = TextConcatFactCheck(fact_checks_path, tasks_path, task_name="monolingual", lang=lang)
    
    df_fc = pd.read_csv(f"scripts/all-MimiLM-L6-v2/processedData/fact_checks_{lang}.csv")
    df_posts_train = pd.read_csv(f"scripts/all-MimiLM-L6-v2/processedData/posts_train_{lang}.csv")
    df_posts_dev = pd.read_csv(f"scripts/all-MimiLM-L6-v2/processedData/posts_dev_{lang}.csv")
    
    if embedings is not None:
        emb_fc = torch.Tensor(embedings[lang]["fc"]).to(device)
        emb_posts_train = torch.Tensor(embedings[lang]["posts_train"]).to(device)
        emb_posts_dev = torch.Tensor(embedings[lang]["posts_dev"]).to(device)
    else:
        sentTransModel = SentenceTransformer(trans_model_name, device=device)
        
        emb_fc = sentTransModel.encode(df_fc["full_text"].values, show_progress_bar=True, normalize_embeddings=True, batch_size=int(128), convert_to_tensor=True)
        emb_posts_train = sentTransModel.encode(df_posts_train["full_text"].values, show_progress_bar=True, normalize_embeddings=True, batch_size=int(128), convert_to_tensor=True)
        emb_posts_dev = sentTransModel.encode(df_posts_dev["full_text"].values, show_progress_bar=True, normalize_embeddings=True, batch_size=int(128), convert_to_tensor=True)
    
    print("Computing semantic search...\n")
    semantic_k = 20
    hits = util.semantic_search(emb_posts_dev, emb_fc, top_k=semantic_k)
    
    # create query by putting in a tuple() the post text and the top 100 fact check text
    queries = []
    fc_top_list = []
    for i in range(len(hits)):
        for j in range(len(hits[i])):
            fc_top_list.append(df_fc.iloc[hits[i][j]["corpus_id"]])
            queries.append((df_posts_dev["full_text"][i], df_fc["full_text"][hits[i][j]["corpus_id"]]))
    df_fc_top = pd.concat(fc_top_list, axis=0)
    
    print("Reranking...")
    crossModel = CrossEncoder(cross_model_name, device=device)
    
    cross_scores = crossModel.predict(queries, show_progress_bar=True, batch_size=128, convert_to_tensor=True, num_workers=16)
    cross_scores = cross_scores.reshape(len(hits), semantic_k)
    max_scores_idx = cross_scores.topk(k=10, dim=1).indices

    # save the predictions to a json file
    d_out[lang] = {df_posts_dev["post_id"].iloc[i]: df_fc_top["fact_check_id"].iloc[max_scores_idx[i].tolist()].tolist() for i in range(len(df_posts_dev))}
    
    
   # save the embeddings to a json file
    emb_out[lang] = {"fc": emb_fc.tolist(), "posts_train": emb_posts_train.tolist(), "posts_dev": emb_posts_dev.tolist()}
    
    
with open("scripts/all-MimiLM-L6-v2/monolingual_predictions.json", "w") as f:
    json.dump(d_out, f)
    
with open("scripts/all-MimiLM-L6-v2/monolingual_embeddings.json", "w") as f:
    json.dump(emb_out, f)

