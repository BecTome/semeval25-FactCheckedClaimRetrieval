from huggingface_hub import HfApi
from tqdm import tqdm
import os
import shutil

def log_info(message):
    import logging

    # Set up logger with timestamp
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logging.info(message)
    

def upload_model_to_hub(local_model_path, hf_model_path, commit_message, exist_ok=True, private=True):
    api = HfApi()

    api.create_repo(hf_model_path, exist_ok=exist_ok, private=private)
    api.upload_folder(
        folder_path=local_model_path,
        repo_id=hf_model_path,
        repo_type="model",
        commit_message=commit_message
    )
    
    log_info(f"Model uploaded to {hf_model_path}")


def cleaning_spacy_cased(text, nlp):
    return " ".join([token.lemma_ for token in nlp(text) if not token.is_stop and not token.is_punct])

def cleaning_spacy(text, nlp):
    return " ".join([token.lemma_.lower() for token in nlp(text) if not token.is_stop and not token.is_punct])

def only_entities(text, nlp):
    return " ".join([ent.text for ent in nlp(text).ents])

def cleaning_spacy_batch(texts, nlp, batch_size=512, n_process=32):
    """
    Processes a batch of texts using spaCy, applying lemmatization, 
    removing stop words, and punctuation.

    Parameters:
    - texts: List of texts to process.
    - nlp: A spaCy language model.

    Returns:
    - List of cleaned and processed texts.
    """
    total = len(texts)
    results = []
    for doc in tqdm(nlp.pipe(texts, disable=["ner", "parser"], batch_size=batch_size, n_process=n_process), total=total):
        results.append(
            " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])
        )
    return results

def copy_directory_contents(src, dst):
    # Ensure the destination directory exists
    if not os.path.exists(dst):
        os.makedirs(dst)
    
    # Loop through each item in the source directory
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dst_item = os.path.join(dst, item)
        
        if os.path.isdir(src_item):
            # Recursively copy subdirectories
            shutil.copytree(src_item, dst_item)
        else:
            # Copy files
            shutil.copy2(src_item, dst_item)