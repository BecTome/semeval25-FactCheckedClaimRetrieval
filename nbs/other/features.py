# %% [markdown]
# # Machine Learning Classifier

# %% [markdown]
# In this notebook, features will be created to train a Machine Learning classifier to predict the similarity between each pair of claims. A simple approach will be used without further tuning.

# %%
# Feature Extraction for Claim Matching

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from dateutil.parser import parse as parse_date
from rapidfuzz import fuzz

from transformers import AutoTokenizer
from sentence_transformers import CrossEncoder
from langdetect import detect
from keybert import KeyBERT


EMB_MODEL_NAME = "/gpfs/projects/bsc14/abecerr1/hub/models--intfloat--multilingual-e5-large/snapshots/ab10c1a7f42e74530fe7ae5be82e6d4f11a719eb"
CE_MODEL_NAME = "/gpfs/projects/bsc14/abecerr1/hub/models--cross-encoder--nli-deberta-v3-large/snapshots/97125691dcb5e470a5ce27b504b8639e3d874433"

# %%
df = pd.read_csv("nbs/other/claim_matching_dataset.csv")
# df = df.fillna("")
df["unverified claim"] = df["unverified claim"].fillna("")
df["reviewed claim"] = df["reviewed claim"].fillna("")

# %% [markdown]
# # Load pretrained models

# %%
# Load NLP model for linguistic features
nlp = spacy.load("xx_ent_wiki_sm")  # multilingual model

# Load sentence embedding model
embedding_model = SentenceTransformer(EMB_MODEL_NAME)

# # Force slow tokenizer
# tokenizer = AutoTokenizer.from_pretrained(CE_MODEL_NAME,)

# Then load the model using CrossEncoder, passing the tokenizer manually
nli_model = CrossEncoder(
    model_name=CE_MODEL_NAME,
    # tokenizer_args={"tokenizer": tokenizer}
)

kw_model = KeyBERT(EMB_MODEL_NAME)


# %%
from tqdm import tqdm
tqdm.pandas()

# df["unverified claim_keywords"] = df["unverified claim"].progress_apply(lambda x: kw_model.extract_keywords(x, keyphrase_ngram_range=(1, 3), stop_words=None))

def create_features(df):
    # --- Feature functions ---
    def compute_cosine_sim(text1, text2):
        if pd.isna(text1) or pd.isna(text2):
            return 0.0
        emb1 = embedding_model.encode(text1, convert_to_tensor=True)
        emb2 = embedding_model.encode(text2, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item()

    def nli_contradiction(claim, evidence):
        return torch.tensor(nli_model.predict([(claim, evidence)])[0]).softmax(dim=0)[0].item()

    def token_overlap(text1, text2):
        tokens1 = set(str(text1).lower().split())
        tokens2 = set(str(text2).lower().split())
        return len(tokens1 & tokens2) / (len(tokens1 | tokens2) + 1e-5)

    def lang_match(text, lang):
        try:
            return int(detect(str(text)) == lang)
        except:
            return 0

    def has_negation(text):
        return int(any(tok.lower_ in ["no", "not", "nunca", "ni", "sin", "n'est", "non"] for tok in nlp(str(text))))

    def shared_named_entities(text1, text2):
        ents1 = set([ent.text.lower() for ent in nlp(str(text1)).ents])
        ents2 = set([ent.text.lower() for ent in nlp(str(text2)).ents])
        return len(ents1 & ents2)
        
    def keyword_overlap(kb_keywords, claim):
            
        try:
            if isinstance(kb_keywords, str):
                keywords = eval(kb_keywords)  # list of (kw, score)
            else:
                keywords = kb_keywords
            
            # print(keywords)
            kw_claim = kw_model.extract_keywords(claim, keyphrase_ngram_range=(1, 3), stop_words=None)
            kw_claim = [kw for kw, _ in kw_claim]
            # print(kw_claim)

            return len([k for k, _ in keywords if k in kw_claim])
        except:
            return 0


    def text_length(text):
        return len(str(text).split())

    def fuzzy_title_match(title, claim):
        return fuzz.partial_ratio(str(title), str(claim)) / 100

    feature_df = pd.DataFrame()

    # print("Extracting keywords...")
    # feature_df["unverified claim_keywords"] = df["unverified claim"].progress_apply(lambda x: kw_model.extract_keywords(x, keyphrase_ngram_range=(1, 3), stop_words=None))
    
    # --- Apply feature engineering ---

    feature_df['ft_cosine_claim_reviewed'] = df.progress_apply(lambda x: compute_cosine_sim(x["unverified claim"], x["reviewed claim"]), axis=1)
    feature_df['ft_cosine_claim_summary'] = df.progress_apply(lambda x: compute_cosine_sim(x["unverified claim"], x["summary"]), axis=1)
    feature_df['ft_cosine_claim_meta_desc'] = df.progress_apply(lambda x: compute_cosine_sim(x["unverified claim"], x["meta_description"]), axis=1)

    # Token overlap and shared named entities
    feature_df['ft_token_overlap'] = df.progress_apply(lambda x: token_overlap(x["unverified claim"], x["reviewed claim"]), axis=1)
    feature_df['ft_shared_named_entities'] = df.progress_apply(lambda x: shared_named_entities(x["unverified claim"], x["reviewed claim"]), axis=1)

    # Boolean flags (easy vectorizations)
    feature_df['ft_has_negation'] = df['unverified claim'].apply(has_negation).astype(int)
    feature_df['ft_has_negation_reviewed'] = df['reviewed claim'].apply(has_negation).astype(int)

    # Keyword overlaps
    print("Computing keyword overlaps...")

    feature_df['ft_keyword_overlap_kb'] = df.progress_apply(lambda x: keyword_overlap(x["kb_keywords"], x["unverified claim"]), axis=1)
    feature_df['ft_keyword_overlap_meta'] = df.progress_apply(lambda x: keyword_overlap(x["meta_keywords"], x["unverified claim"]), axis=1)

    # Domain match (fully vectorized)
    feature_df['ft_domain_match'] = ((df['url'].notna()) & (df.apply(lambda row: row['domain'] in row['url'], axis=1))).astype(int)

    # Language match and image/video presence
    feature_df['ft_lang_match'] = df.progress_apply(lambda x: lang_match(x["unverified claim"], x["meta_lang"]), axis=1)
    feature_df['ft_has_image'] = df['cr_image'].notna().astype(int)
    feature_df['ft_has_video'] = df['movies'].apply(lambda x: int(len(eval(x)) > 0) if pd.notna(x) else 0)

    # Text lengths
    feature_df['ft_text_len_summary'] = df['summary'].apply(text_length)
    feature_df['ft_text_len_meta'] = df['meta_description'].apply(text_length)

    # Fuzzy matching
    feature_df['ft_title_match_ratio'] = df.progress_apply(lambda x: fuzzy_title_match(x["title"], x["unverified claim"]), axis=1)


    nli_probs = nli_model.predict(df[["unverified claim", "reviewed claim"]].fillna("").values.tolist())
    nli_probs = torch.tensor(nli_probs).softmax(dim=1).numpy()
    feature_df["ft_contr_nli_claim_reviewed"] = nli_probs[:, 0]

    nli_probs = nli_model.predict(df[["unverified claim", "summary"]].fillna("").values.tolist())
    nli_probs = torch.tensor(nli_probs).softmax(dim=1).numpy()
    feature_df["ft_contr_nli_claim_summary"] = nli_probs[:, 0]

    nli_probs = nli_model.predict(df[["unverified claim", "meta_description"]].fillna("").values.tolist())
    nli_probs = torch.tensor(nli_probs).softmax(dim=1).numpy()
    feature_df["ft_contr_nli_claim_meta_desc"] = nli_probs[:, 0]

    return feature_df


# %%
df_features = create_features(df)
df_features["similarity"] = df["similarity"]
df_features.to_csv("nbs/other/claim_matching_features.csv", index=False)