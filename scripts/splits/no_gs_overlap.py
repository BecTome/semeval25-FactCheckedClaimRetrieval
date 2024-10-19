import os
import json
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.datasets import TextConcatPosts
from src import config
from src.splits import get_split_indices_no_gs_overlap
from src.utils import log_info

# Paths and parameters
original_tasks_path = config.PHASE1_TASKS_PATH
posts_path = config.POSTS_PATH
fact_checks_path = config.FACT_CHECKS_PATH
gs_path = config.GS_PATH

test_size = 0.1
random_state = 42
experiment_name = "no_gs_overlap"
splits_path = "data/splits/"

# Step 1: Create splits directory if it doesn't exist
if not os.path.exists(splits_path):
    os.makedirs(splits_path)
    log_info(f"Created splits directory at {splits_path}")

# Step 2: Crosslingual processing
log_info("Starting crosslingual processing...")
posts_xl = TextConcatPosts(posts_path, original_tasks_path, task_name="crosslingual", gs_path=gs_path)
idx_train, idxs_dev = get_split_indices_no_gs_overlap(posts_xl.df_train, test_size=test_size, random_state=random_state)

new_tasks_xl = posts_xl.tasks["crosslingual"].copy()
new_tasks_mono = posts_xl.tasks["monolingual"].copy()

new_tasks_xl["posts_train"] = idx_train
new_tasks_xl["posts_dev"] = idxs_dev
log_info("Crosslingual split created successfully.")

# Step 3: Monolingual processing
log_info("Starting monolingual processing for each language...")
for lan in tqdm(new_tasks_mono.keys()):
    log_info(f"Processing monolingual data for language: {lan}")
    posts = TextConcatPosts(posts_path, original_tasks_path, task_name="monolingual", gs_path=gs_path, lang=lan)
    df_posts_train = posts.df_train

    idx_train, idxs_dev = get_split_indices_no_gs_overlap(df_posts_train, test_size=test_size, random_state=random_state)
    
    new_tasks_mono[lan]["posts_train"] = idx_train
    new_tasks_mono[lan]["posts_dev"] = idxs_dev
    log_info(f"Monolingual split for {lan} completed.")

# Combine new tasks
new_tasks = {
    "monolingual": new_tasks_mono,
    "crosslingual": new_tasks_xl
}
log_info("Combined monolingual and crosslingual splits.")

# Step 4: Create statistics
log_info("Generating statistics for the splits...")
d_stats = {}
for lan in new_tasks["monolingual"].keys():
    d_stats[lan] = {
        "train": len(new_tasks["monolingual"][lan]["posts_train"]),
        "dev": len(new_tasks["monolingual"][lan]["posts_dev"]),
        "ratio": len(new_tasks["monolingual"][lan]["posts_dev"]) / len(new_tasks["monolingual"][lan]["posts_train"])
    }

d_stats["crosslingual"] = {
    "train": len(new_tasks["crosslingual"]["posts_train"]),
    "dev": len(new_tasks["crosslingual"]["posts_dev"]),
    "ratio": len(new_tasks["crosslingual"]["posts_dev"]) / len(new_tasks["crosslingual"]["posts_train"])
}

df_stats = pd.DataFrame(d_stats).T
log_info("Statistics generated.")

# Step 5: Save split and statistics
log_info("Saving new tasks and statistics to disk...")
new_tasks_path = os.path.join(splits_path, f"tasks_{experiment_name}.json")
with open(new_tasks_path, "w") as f:
    json.dump(new_tasks, f)
log_info(f"Saved tasks split to {new_tasks_path}.")

# Save statistics
stats_path = os.path.join(splits_path, "stats")
if not os.path.exists(stats_path):
    os.makedirs(stats_path)
    log_info(f"Created stats directory at {stats_path}")

df_stats.to_csv(os.path.join(stats_path, f"tasks_{experiment_name}.csv"), index=True)
log_info(f"Saved statistics to {stats_path}/tasks_{experiment_name}.csv.")
