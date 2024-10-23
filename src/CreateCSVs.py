import pandas as pd

from tqdm import tqdm

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from datasets import TextConcatFactCheck, TextConcatPosts

tasks_path = "data/splits/tasks_no_gs_overlap.json"
posts_path = "data/complete_data/posts.csv"
fact_checks_path = "data/complete_data/fact_checks.csv"
out_path = "scripts/e5_and_CrossEncoder/processedData/"
gs_path = "data/complete_data/pairs.csv"
langs = ['fra', 'spa', 'eng', 'por', 'tha', 'deu', 'msa', 'ara']

for lang in tqdm(langs, desc="Languages", ):
    
    posts = TextConcatPosts(posts_path, tasks_path, task_name="monolingual", gs_path=gs_path, lang=lang, prefix="query: ")
    fact_checks = TextConcatFactCheck(fact_checks_path, tasks_path, task_name="monolingual", lang=lang, prefix="passage: ")
        
    fact_checks.df.to_csv(f"{out_path}fact_checks_{lang}.csv")
    posts.df_train.to_csv(f"{out_path}posts_train_{lang}.csv")
    posts.df_dev.to_csv(f"{out_path}posts_dev_{lang}.csv")
        