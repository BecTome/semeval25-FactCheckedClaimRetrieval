# %%

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src import config
from src.datasets import TextConcatFactCheck, TextConcatPosts

# Paths from config
posts_path = config.POSTS_PATH
fact_checks_path = config.FACT_CHECKS_PATH
gs_path = config.GS_PATH
task_path = config.TASKS_PATH
task_name = "monolingual"
lang = "eng"

posts = TextConcatPosts(posts_path, tasks_path=task_path, gs_path=gs_path, task_name=task_name, lang=lang)
fcs = TextConcatFactCheck(fact_checks_path, tasks_path=task_path, task_name=task_name, lang=lang)

# df_posts_train = posts.df_train
df_posts_dev = posts.df_dev
df_fc = fcs.df

# %%
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import numpy as np
from abc import abstractmethod
import json
import os
from src.models import BaseModel

class CrossencoderModel(BaseModel):
    def __init__(self, model_name, df_cands, df_fc, device="cuda", show_progress_bar=True, batch_size=128, k=10, **kwargs):
        self.model = CrossEncoder(model_name, device=device, **kwargs)
        self.idx_to_text = df_fc["full_text"].to_dict()
        self.df_cands = df_cands.copy()
        self.vectorized_map = np.vectorize(lambda x: self.idx_to_text.get(x, None))        
        # self.normalize_embeddings = normalize_embeddings
        # self.emb_fc = self.encode(df_fc["full_text"].values)
        # self.pos_to_idx = {pos: idx for pos, idx in enumerate(df_fc.index)}
        super().__init__(device, show_progress_bar, batch_size, k)


    # def encode(self, texts):
    #     return torch.tensor(self.model.encode(texts, device=self.device, show_progress_bar=self.show_progress_bar, 
    #                                           batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings))
        
    def train(self, texts):
        pass
    
    def predict(self, post, cands_list):
        cands_list_text = self.vectorized_map(cands_list)
        pos_ids = self.model.rank(post, cands_list_text, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, top_k=self.k, convert_to_numpy=True)
        pos_ids = [pos_id["corpus_id"] for pos_id in pos_ids]
        return np.array(cands_list)[pos_ids]

# %%
from src.models import EmbeddingModel

biencoder_name = "/gpfs/projects/bsc14/abecerr1/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/ae06c001a2546bef168b9bf8f570ccb1a16aaa27"
cand_ret_model = EmbeddingModel(biencoder_name, df_fc, show_progress_bar=True, batch_size=256, normalize_embeddings=True, k=100)

df_posts_dev["preds"] = cand_ret_model.predict(df_posts_dev["full_text"].values).tolist()



# %%
cross_model = CrossencoderModel("cross-encoder/ms-marco-MiniLM-L-6-v2", df_posts_dev, df_fc, show_progress_bar=False, batch_size=512, k=10)

df_posts_dev["preds_cross"] = df_posts_dev.apply(lambda x:cross_model.predict(x["full_text"], x["preds"]), axis=1)

df_posts_dev_2 = df_posts_dev.copy()
df_posts_dev_2["preds"] = df_posts_dev_2["preds_cross"]

print(cross_model.evaluate(df_posts_dev_2, task_name=task_name, lang=lang))
