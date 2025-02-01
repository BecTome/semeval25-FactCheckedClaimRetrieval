#!/usr/bin/env python
import argparse
from tqdm import tqdm
import json
import sys
import os
from datetime import datetime
from time import time
import pandas as pd
import torch
pd.set_option('display.max_columns', None)

# Importing from src
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.datasets import TextConcatFactCheck, TextConcatPosts
from src.models import EmbeddingModel
from src import config
from src.utils import log_info

def run_task(tasks_path, task_name, langs, model_name, output_path):
    """
    Run the task with the given parameters.
    """
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    if output_path is not None:
        output_path = os.path.join(output_path, task_name, current_time)
    
    langs = ["eng"] if task_name == "crosslingual" else langs
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    log_info(f"Task: {task_name}")
    log_info(f"Tasks path: {tasks_path}")
    log_info(f"Languages: {langs}")
    log_info(f"Model: {model_name}")
    log_info(f"Output path: {output_path}")
    log_info(f"Device: {device}\n")

    # Paths from config
    posts_path = config.POSTS_PATH
    fact_checks_path = config.FACT_CHECKS_PATH
    gs_path = config.GS_PATH

    # tasks_path = "data/splits/tasks_local_dev.json"
    ls_k = [1, 3, 5, 10]
    
    d_out = {}
    df_eval = pd.DataFrame(index=ls_k)
    df_eval.index.name = "k"
    
    for lang in tqdm(langs, desc="Languages"):
        log_info(f"Lang: {lang}")
        time_start_lang = time()
        
        log_info("Loading posts...")
        time_start = time()
        posts = TextConcatPosts(posts_path, tasks_path, task_name=task_name, gs_path=gs_path, lang=lang, version="english")
        log_info(f"Loaded {len(posts)}")
        log_info(f"Time taken: {time() - time_start:.2f}s\n")
        
        log_info("Loading fact checks..")
        time_start = time()
        fact_checks = TextConcatFactCheck(fact_checks_path, tasks_path, task_name=task_name, lang=lang, version="english")
        log_info(f"Loaded {len(fact_checks)}")
        log_info(f"Time taken: {time() - time_start:.2f}s\n")

        df_fc = fact_checks.df
        # df_posts_train = posts.df_train
        df_posts_dev = posts.df_dev
        log_info("Loading model...")
        time_start = time()
        model = EmbeddingModel(model_name, df_fc, batch_size=256, model_type=None, device=device, prompt="Represent the FactCheck sentences for retrieval")
        log_info(f"Time taken: {time() - time_start:.2f}s\n")
        
        log_info("Predicting...")
        time_start = time()
        df_posts_dev["preds"] = model.predict(df_posts_dev["full_text"].values, prompt="Represent the Social Media posts text for retrieving FactChecks").tolist()
        log_info(f"Time taken: {time() - time_start:.2f}s\n")
        
        d_out.update(df_posts_dev["preds"].to_dict())

        log_info("Evaluating...")
        log_info(f"Dev shape: {df_posts_dev.shape}")
        time_start = time()
        d_eval_i = model.evaluate(df_posts_dev, task_name=task_name, lang=lang, output_folder=output_path)
        
        log_info(f"Time taken: {time() - time_start:.2f}s\n")
        
        log_info(f"Evaluation results success@k")
        df_eval = pd.concat([df_eval, pd.DataFrame(d_eval_i[task_name])], axis=1)
        
        log_info(f"\n{df_eval}")
        
        # Display the DataFrame
        # log_info("\n\n")
        
        log_info(f"\nTime taken for lang {lang}: {time() - time_start_lang:.2f}s\n")
    
    df_eval["avg"] = df_eval.mean(axis=1)

    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, f"{task_name}_predictions.json"), "w") as f:
            json.dump(d_out, f)
            
        df_eval.to_csv(os.path.join(output_path, f"{task_name}_evaluation.csv"))
        
    return df_eval, d_out

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Run embedding prediction task.')
    parser.add_argument('--task_name', type=str, required=True, help="Choose 'monolingual' or 'crosslingual'")
    parser.add_argument('--model_name', type=str, required=True, help="Path to the model")
    
    parser.add_argument('--output_path', type=str, default=None, help="Directory to save output")
    parser.add_argument('--task_file', type=str, default=config.TASKS_PATH, help="Path to the task file")
    parser.add_argument('--langs', type=str, nargs='+', default=config.LANGS, help="List of languages")

    args = parser.parse_args()

    run_task(args.task_file, args.task_name, args.langs, args.model_name, args.output_path)

if __name__ == "__main__":
    main()
