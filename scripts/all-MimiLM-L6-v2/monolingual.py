#!/usr/bin/env python
from tqdm import tqdm
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.datasets import TextConcatFactCheck, TextConcatPosts
from src.models import EmbeddingModel

tasks_path = "data/complete_data/tasks.json"
posts_path = "data/complete_data/posts.csv"
fact_checks_path = "data/complete_data/fact_checks.csv"
gs_path = "data/complete_data/pairs.csv"
langs = ['fra', 'spa', 'eng', 'por', 'tha', 'deu', 'msa', 'ara']
model_name = 'sentence-transformers/all-MiniLM-L6-v2'

d_out = {}
for lang in tqdm(langs, desc="Languages"):

    posts = TextConcatPosts(posts_path, tasks_path, task_name="monolingual", gs_path=gs_path, lang=lang)
    fact_checks = TextConcatFactCheck(fact_checks_path, tasks_path, task_name="monolingual", lang=lang)

    df_fc = fact_checks.df
    df_posts_train = posts.df_train
    df_posts_dev = posts.df_dev

    model = EmbeddingModel(model_name, df_fc)

    df_posts_dev["preds"] = model.predict(df_posts_dev["full_text"].values).tolist()
    d_out.update(df_posts_dev["preds"].to_dict())

with open("data/out/all-MiniLM-L6-v2/monolingual_predictions.json", "w") as f:
    json.dump(d_out, f)

