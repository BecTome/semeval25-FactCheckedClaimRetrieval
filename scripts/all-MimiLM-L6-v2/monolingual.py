#!/usr/bin/env python

from SimpleITK import namedtuple
from numpy import cross
import pandas as pd

from sentence_transformers import SentenceTransformer, CrossEncoder, util

from tqdm import tqdm
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.datasets import TextConcatFactCheck, TextConcatPosts
from src.models import EmbeddingModel

tasks_path = "data/splits/tasks_no_gs_overlap.json"
posts_path = "data/complete_data/posts.csv"
fact_checks_path = "data/complete_data/fact_checks.csv"
gs_path = "data/complete_data/pairs.csv"
langs = ['fra', 'spa', 'eng', 'por', 'tha', 'deu', 'msa', 'ara']
trans_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
cross_model_name = 'cross-encoder/ms-marco-MiniLM-L6-v2'

d_out = {}
emb_out = {}
for lang in tqdm(langs, desc="Languages"):

    posts = TextConcatPosts(posts_path, tasks_path, task_name="monolingual", gs_path=gs_path, lang=lang)
    fact_checks = TextConcatFactCheck(fact_checks_path, tasks_path, task_name="monolingual", lang=lang)
    
    df_fc = pd.read_csv(f"scripts/all-MimiLM-L6-v2/processedData/fact_checks_{lang}.csv")
    df_posts_train = pd.read_csv(f"scripts/all-MimiLM-L6-v2/processedData/posts_train_{lang}.csv")
    df_posts_dev = pd.read_csv(f"scripts/all-MimiLM-L6-v2/processedData/posts_dev_{lang}.csv")

    sentTransModel = SentenceTransformer(trans_model_name, device="cuda")
    
    crossModel = CrossEncoder(cross_model_name, device="cuda")
    
    emb_fc = sentTransModel.encode(df_fc["full_text"].values, show_progress_bar=True, normalize_embeddings=True, batch_size=128, convert_to_tensor=True)
    emb_posts_train = sentTransModel.encode(df_posts_train["text"].values, show_progress_bar=True, normalize_embeddings=True, batch_size=128, convert_to_tensor=True)
    emb_posts_dev = sentTransModel.encode(df_posts_dev["full_text"].values, show_progress_bar=True, normalize_embeddings=True, batch_size=128, convert_to_tensor=True)
    
    # emb_fc = torch.randn((len(df_fc), 384))
    # emb_posts = torch.randn((len(df_posts_dev), 384))
    
    for n, post in enumerate(df_posts_dev["text"].values):
        similarity_scores = sentTransModel.similarity(emb_posts_dev[n], emb_fc)
        scores, indices = similarity_scores.topk(50)
        
        # Now, score all retrieved passages with the cross_encoder
        cross_inp = [[post, df_fc["full_text"][idx]] for idx in indices]
        cross_scores = crossModel.predict(cross_inp)
        scores, indices = cross_scores.topk(10)
        # indices = torch.randint(0, len(df_fc)-1, (10,)).tolist()

        d_out.update({df_posts_dev.index[n]: df_fc.index[indices].tolist()})
        
        
    #save the embeddings to a json file
    emb_out[lang] = {"fc": emb_fc.tolist(), "posts_train": emb_posts_train.tolist(), "posts_dev": emb_posts_dev.tolist()}
    
    
with open("scripts/all-MimiLM-L6-v2/monolingual_predictions.json", "w") as f:
    json.dump(d_out, f)
    
with open("scripts/all-MimiLM-L6-v2/monolingual_embeddings.json", "w") as f:
    json.dump(emb_out, f)

