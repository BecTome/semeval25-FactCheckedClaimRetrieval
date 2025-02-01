import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import torch
import concurrent.futures
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, util
from src.datasets import TextConcatFactCheck, TextConcatPosts
from src import config
import json

# Add root directory to sys path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Paths and configurations
tasks_path = config.TASKS_PATH
posts_path = config.POSTS_PATH
fact_checks_path = config.FACT_CHECKS_PATH
gs_path = config.GS_PATH
langs = config.LANGS
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the embedding model once, shared across all threads
embed_model = SentenceTransformer(config.E5ENCODER, device=device)

# Function to process each language and generate training samples
def process_language(lang):
    print(f"Processing language: {lang}")
    train_samples = []
    
    # Load posts and fact-checks
    posts = TextConcatPosts(posts_path, tasks_path, task_name="monolingual", gs_path=gs_path, lang=lang, version="original")
    fact_checks = TextConcatFactCheck(fact_checks_path, tasks_path, task_name="monolingual", lang=lang, version="original")
    df_fc = fact_checks.df
    df_posts_train = posts.df_train

    for _, row in df_posts_train.iterrows():
        fc_ids = row['gs']
        pos_text = row['full_text']

        # Positive samples
        pos_texts = [df_fc.loc[fc_id]['full_text'] for fc_id in fc_ids]
        train_samples.extend([InputExample(texts=[pos_text, text], label=1.0) for text in pos_texts])

        # Negative samples
        mask = ~df_fc.index.isin(fc_ids)
        negative_samples = df_fc[mask].sample(n=3)
        
        # Batch encode to minimize calls
        neg_texts = negative_samples['full_text'].tolist()
        batch_texts = [pos_text] + neg_texts
        embeddings = embed_model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)

        # Compute cosine similarities for negatives
        pos_emb = embeddings[0]
        for n,neg_emb in enumerate(embeddings[1:]):
            similarity = util.pytorch_cos_sim(pos_emb, neg_emb).item()
            train_samples.append(InputExample(texts=[pos_text, neg_texts[n]], label=similarity))

    print(f"Training samples created for language: {lang}")
    return train_samples

# Parallel processing
all_train_samples = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_language, lang) for lang in langs]
    for future in concurrent.futures.as_completed(futures):
        all_train_samples.extend(future.result())

print(f"Total training samples: {len(all_train_samples)}")
print(all_train_samples[0])  # Verify the first training sample

# Convert InputExample objects to JSON-serializable dictionaries
samples_data = [{
    "texts": sample.texts,
    "label": sample.label
} for sample in all_train_samples]

# Save as JSON file
with open("scritps/trainCrossEncoder/train_samples.json", "w") as f:
    json.dump(samples_data, f)