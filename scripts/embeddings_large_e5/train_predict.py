#!/usr/bin/env python
from tqdm import tqdm
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.datasets import TextConcatFactCheck, TextConcatPosts
from src.models import EmbeddingModel
from src import config

from datetime import datetime
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

posts_path = config.POSTS_PATH
fact_checks_path = config.FACT_CHECKS_PATH
gs_path = config.GS_PATH
output_path = config.OUTPUT_PATH

tasks_path = "data/splits/tasks_local_dev.json"
task_name = "crosslingual"                                          # Choose monolingual or crosslingual
langs = ['fra', 'spa', 'eng', 'por', 'tha', 'deu', 'msa', 'ara']
langs = langs if task_name == "monolingual" else ["eng"]
model_name = '/home/bsc/bsc830651/.cache/huggingface/hub/models--intfloat--multilingual-e5-large/snapshots/ab10c1a7f42e74530fe7ae5be82e6d4f11a719eb'
output_path = "output/embeddings_large_e5"
output_path = os.path.join(output_path, task_name, current_time)

print(f"Task: {task_name}")
print(f"Languages: {langs}")
print(f"Model: {model_name}")
print(f"Output path: {output_path}")

    
d_out = {}
for lang in tqdm(langs, desc="Languages"):

    posts = TextConcatPosts(posts_path, tasks_path, task_name=task_name, gs_path=gs_path, lang=lang)
    fact_checks = TextConcatFactCheck(fact_checks_path, tasks_path, task_name=task_name, lang=lang)

    df_fc = fact_checks.df
    df_posts_train = posts.df_train
    df_posts_dev = posts.df_dev

    model = EmbeddingModel(model_name, df_fc)

    df_posts_dev["preds"] = model.predict(df_posts_dev["full_text"].values).tolist()
    d_out.update(df_posts_dev["preds"].to_dict())


if not os.path.exists(output_path):
    os.makedirs(output_path)
    
with open(os.path.join(output_path, f"{task_name}_predictions.json"), "w") as f:
    json.dump(d_out, f)

