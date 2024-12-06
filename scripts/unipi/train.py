#!/usr/bin/env python
import argparse
from tqdm import tqdm
tqdm.pandas()

import json
import sys
import os
from datetime import datetime
from time import time
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader
from torch import nn
from sentence_transformers.readers import InputExample

from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sklearn.model_selection import train_test_split
import math

# Importing from src
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.datasets import TextConcatFactCheck, TextConcatPosts
from src.models import IEModel, EmbeddingModel
from src import config
from src.utils import log_info
from src.contrastive import generate_triplets_candidates


def run_task(tasks_path, task_name, langs, teacher_model_name, reranker_model_name, output_path, model_save_path, triplets_path=None, d_config={}):
    """
    Run the task with the given parameters.
    """
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    if output_path is not None:
        output_path = os.path.join(output_path, task_name, current_time)
    
    langs = ["eng"] if task_name == "crosslingual" else langs
    
    log_info(f"Task: {task_name}")
    log_info(f"Tasks path: {tasks_path}")
    log_info(f"Languages: {langs}")
    log_info(f"Teacher Model: {teacher_model_name}")
    log_info(f"Reranker Model: {reranker_model_name}")
    log_info(f"Output path: {output_path}\n")
    log_info(f"Triplets path: {triplets_path}\n")

    # Paths from config
    posts_path = config.POSTS_PATH
    fact_checks_path = config.FACT_CHECKS_PATH
    gs_path = config.GS_PATH
    
    train_batch_size = d_config.get("train_batch_size", 128)            # Dataloader batch size (NO NEED TO TUNE)
    num_epochs = d_config.get("num_epochs", 10)                         # Number of epochs (TUNEABLE PARAMETER. MONITOR LOSS)
    dev_size_triplets = d_config.get("dev_size_triplets", 0.1)          # Dev size for triplets (NO NEED TO TUNE)
    pct_warmup = d_config.get("pct_warmup", 0.1)                        # Percentage of warmup steps (PARTLY TUNEABLE)
    output_k = d_config.get("output_k", 10)                             # Number of outputs to consider (NO NEED TO TUNE)
    optimizer_params = d_config.get("optimizer_params", {"lr": 2e-5})   # Optimizer parameters (TUNEABLE PARAMETERS. EXPLORE MORE)
    emb_batch_size = d_config.get("emb_batch_size", 256)                # Batch size for embedding model (NO NEED TO TUNE)
    n_candidates = d_config.get("n_candidates", 100)                    # Candidates to consider for reranking (TUNEABLE PARAMETER)
    n_neg_candidates = d_config.get("n_neg_candidates", 4)              # Number of negative candidates to consider (TUNEABLE PARAMETER. BEWARE OF UNBALANCE AND METRIC)
    neg_perc_threshold = d_config.get("neg_perc_threshold", 0.9)        # Negative percentage threshold. Only consider as negatives candidates with a score below 
                                                                        # neg_perc_threshold * postive_score (TUNEABLE PARAMETER)
                                    
    log_info(f"Train batch size: {train_batch_size}")
    log_info(f"Num epochs: {num_epochs}")
    log_info(f"Dev size triplets: {dev_size_triplets}")
    log_info(f"Percentage warmup steps: {pct_warmup}")
    log_info(f"Output k: {output_k}")
    log_info(f"Optimizer params: {optimizer_params}")
    log_info(f"Embedding batch size: {emb_batch_size}")
    log_info(f"Number of candidates: {n_candidates}")
    log_info(f"Number of negative candidates: {n_neg_candidates}")
    # log_info(f"Negative percentage threshold: {neg_perc_threshold}\n")

    
    # tasks_path = "data/splits/tasks_local_dev.json"
    ls_k = [1, 3, 5, 10]
    
    d_out = {}
    df_eval = pd.DataFrame(index=ls_k)
    df_eval.index.name = "k"
    
    for lang in tqdm(langs, desc="Languages"):

        if model_save_path is not None:
            model_save_path = os.path.join(model_save_path, f"{task_name}_{lang}_reranker")
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)


        log_info(f"Lang: {lang}")
        time_start_lang = time()
        
        log_info("Loading posts...")
        time_start = time()
        posts = TextConcatPosts(posts_path, tasks_path, task_name=task_name, gs_path=gs_path, lang=lang)
        posts_en = TextConcatPosts(posts_path, tasks_path, task_name=task_name, gs_path=gs_path, lang=lang, version="english")
        log_info(f"Loaded {len(posts)}")
        log_info(f"Time taken: {time() - time_start:.2f}s\n")
        
        log_info("Loading fact checks..")
        time_start = time()
        fact_checks = TextConcatFactCheck(fact_checks_path, tasks_path, task_name=task_name, lang=lang)
        fact_checks_en = TextConcatFactCheck(fact_checks_path, tasks_path, task_name=task_name, lang=lang, version="english")
        log_info(f"Loaded {len(fact_checks)}")
        log_info(f"Time taken: {time() - time_start:.2f}s\n")

        df_fc = fact_checks.df
        df_fc_en = fact_checks_en.df
        df_posts_train = posts.df_train
        df_posts_en = posts_en.df_train
        df_posts_dev = posts.df_dev
    
        log_info(f"Loading Teacher Model: {teacher_model_name}...")
        time_start = time()
        teacher_model = EmbeddingModel(teacher_model_name, df_fc, batch_size=emb_batch_size)
        log_info(f"Time taken Loading Teacher Model: {time() - time_start:.2f}s\n")

        candidate_model = IEModel("en_core_web_trf", df_fc_en, k=None)
        log_info(f"Generating candidates...")
        time_start = time()
        df_posts_train["candidates"] = candidate_model.predict(df_posts_en["full_text"].values)
        log_info(f"Time taken CANDIDATES GENERATION: {time() - time_start:.2f}s\n")
        
        log_info(f"Generating triplets...")
        time_start = time()
        if triplets_path is not None:
            df_cl = pd.read_csv(triplets_path)
            log_info(f"Loaded {len(df_cl)} triplets from {triplets_path} in {time() - time_start:.2f}s\n")
        else:
            df_cl = generate_triplets_candidates(df_posts_train, df_fc, teacher_model, n_candidates=n_neg_candidates, neg_perc_threshold=neg_perc_threshold)
            log_info(f"Time taken TRIPLETS GENERATION: {time() - time_start:.2f}s\n")

        if model_save_path is not None:
            df_cl.to_csv(os.path.join(model_save_path, f"triplets.csv"), index=False)
        
        rerank_model = CrossEncoder(reranker_model_name, num_labels=1, max_length=1024, trust_remote_code=True)

        train_samples = df_cl.progress_apply(lambda x: InputExample(texts=[x["query"], x["passage"]], label=x["label"]), axis=1).tolist()
        
        # Split the dataset into train and dev
        train_samples, dev_samples = train_test_split(train_samples, test_size=dev_size_triplets, random_state=42)

        # We wrap train_samples, which is a list of InputExample, in a pytorch DataLoader
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

        accuracy_evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name=f"{task_name}_{lang}")
        
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * pct_warmup)  # 10% of train data for warm-up

        def pearson_corr(x, y):
            import torch
            # Center inputs by subtracting means
            x = x - x.mean()
            y = y - y.mean()
            
            # Compute correlation
            corr = (x * y).sum() / (torch.sqrt((x**2).sum()) * torch.sqrt((y**2).sum()))
            return corr

        pearson_loss = lambda x, y: -pearson_corr(x, y)
        sigmoid = nn.Sigmoid()
        
        # Train the model
        rerank_model.fit(
                            train_dataloader=train_dataloader,
                            evaluator=accuracy_evaluator,
                            epochs=num_epochs,
                            evaluation_steps=10000,
                            warmup_steps=warmup_steps,
                            output_path=model_save_path,
                            activation_fct=sigmoid,          # Activation function is sigmoid because otherwise logits are returned
                            optimizer_params=optimizer_params,
                            loss_fct=pearson_loss
                        )
        
        if model_save_path is not None:
            log_info(f"Model saved to: {model_save_path}")
                        
        log_info("Predicting...")
        time_start = time()
        arr_cands = teacher_model.predict(df_posts_dev["full_text"].values)[:,:n_candidates]
        
        # df_dev_out = df_posts_dev[["full_text", "gs"]].copy()
        ls_preds = []
        for i in tqdm(range(len(df_posts_dev))):
            ranked_vals = rerank_model.rank(df_posts_dev["full_text"].values[i], df_fc.loc[arr_cands[i], "full_text"].values, show_progress_bar=False)
            ls_preds.append([int(arr_cands[i, dd["corpus_id"]]) for dd in ranked_vals][:output_k])
        
        df_posts_dev["preds"] = ls_preds
        # df_posts_dev["preds"] = model.predict(df_posts_dev["full_text"].values).tolist()
        log_info(f"Time taken: {time() - time_start:.2f}s\n")
        
        d_out.update(df_posts_dev["preds"].to_dict())
        
        log_info("Evaluating...")
        log_info(f"Dev shape: {df_posts_dev.shape}")
        time_start = time()
        d_eval_i = teacher_model.evaluate(df_posts_dev, task_name=task_name, lang=lang, output_folder=output_path)
        
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
    parser.add_argument('--teacher_model_name', type=str, required=True, help="Path to the teacher model")
    parser.add_argument('--reranker_model_name', type=str, required=True, help="Path to the reranker model")
    parser.add_argument('--triplets_path', type=str, default=None, help="Path to the triplets file")
    
    parser.add_argument('--output_path', type=str, default=None, help="Directory to save output")
    parser.add_argument('--model_save_path', type=str, default=None, help="Directory to save model")
    parser.add_argument('--task_file', type=str, default=config.TASKS_PATH, help="Path to the task file")
    parser.add_argument('--langs', type=str, nargs='+', default=config.LANGS, help="List of languages")

    args = parser.parse_args()

    run_task(args.task_file, args.task_name, args.langs, args.teacher_model_name, args.reranker_model_name, args.output_path, args.model_save_path)

if __name__ == "__main__":
    main()
