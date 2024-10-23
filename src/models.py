from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import numpy as np
from abc import abstractmethod
import json
import os

class BaseModel:
    def __init__(self, device="cuda", show_progress_bar=True, batch_size=128, k=10):
        self.device = device
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.k = k
    
    @abstractmethod
    def train(self, texts):
        pass
    
    @abstractmethod
    def predict(self, texts1, texts2):
        pass
    
    def evaluate(self, df_eval, task_name, lang, ls_k=[1, 3, 5, 10], output_folder=None):
        """
        Evaluate the predictions based on the ground truth.
        Inputs:
        df_eval: DataFrame with the predictions in a column named "preds"
        ls_k: List of k values to evaluate
        output_folder: Folder to save the evaluation results (default: None)
        
        Returns:
        d_eval: Dictionary with the evaluation results
        """
        assert task_name in ["monolingual", "crosslingual"], "task_name should be either monolingual or crosslingual"

        if task_name == "crosslingual":
            lang = f"eng"
            
        # d_eval = {task_name: {lang: {"group": {}, "individual": {}}}}
        d_eval = {task_name: {lang: {}}}

        if df_eval is None:
            df_eval = self.df.copy()
            
        for k in ls_k:
            d_eval[task_name][lang][k] = df_eval.apply(lambda x: len(list((set(x["preds"][:k]) & set(x["gs"])))) > 0, axis=1).mean()
            # d_eval[task_name][lang]["individual"][k] = df_eval.explode("gs").apply(lambda x: x["gs"] in x["preds"][:k], axis=1).mean()

        # if output_folder is not None:
        #     eval_filename = "evaluation.json"
        #     output_file = os.path.join(output_folder, eval_filename)
        #     json.dump(d_eval, open(output_file, "w"), indent=4)
            
        return d_eval

class EmbeddingModel(BaseModel):
    def __init__(self, model_name, df_fc, device="cuda", show_progress_bar=True, batch_size=128, normalize_embeddings=True, k=10):
        super().__init__(device, show_progress_bar, batch_size, k)
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize_embeddings = normalize_embeddings
        self.emb_fc = self.encode(df_fc["full_text"].values)
        self.pos_to_idx = {pos: idx for pos, idx in enumerate(df_fc.index)}


    def encode(self, texts):
        return torch.tensor(self.model.encode(texts, show_progress_bar=self.show_progress_bar, 
                                              batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings)) #  device=self.device,
        
    def train(self, texts):
        pass
    
    def similarity(self, emb1, emb2):
        return torch.mm(emb1, emb2.T).cpu().numpy()
    
    def predict(self, texts):
        arr1 = self.encode(texts)
        sim = self.similarity(arr1, self.emb_fc)
        idx_sim = np.argsort(sim, axis=1)[:, ::-1][:, :self.k]
        # Apply the function element-wise to the array
        vectorized_map = np.vectorize(lambda x: self.pos_to_idx.get(x, None))
        return vectorized_map(idx_sim)
    

class CrossencoderModel(BaseModel):
    '''
    Given a text and a list of candidate positions (cands_list), this model ranks the candidates based on the similarity with the text.
    Returns the top-k candidates in form of fact_check_ids
    '''
    def __init__(self, model_name, df_fc, device="cuda", show_progress_bar=True, batch_size=128, k=10, **kwargs):
        self.model = CrossEncoder(model_name, device=device, **kwargs)
        self.idx_to_text = df_fc["full_text"].to_dict()
        self.vectorized_map = np.vectorize(lambda x: self.idx_to_text.get(x, None))        
        super().__init__(device, show_progress_bar, batch_size, k)

    def train(self, texts):
        pass
    
    def predict(self, post, cands_list):
        # df_posts_dev["preds"] = df_posts_dev.apply(lambda x:cross_model.predict(x["full_text"], x["preds"]), axis=1)
        cands_list_text = self.vectorized_map(cands_list)
        pos_ids = self.model.rank(post, cands_list_text, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, top_k=self.k, convert_to_numpy=True)
        pos_ids = [pos_id["corpus_id"] for pos_id in pos_ids]
        return np.array(cands_list)[pos_ids]
