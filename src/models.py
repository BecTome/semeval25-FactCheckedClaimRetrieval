from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import numpy as np
from abc import abstractmethod
import json
import os
import spacy
from tqdm import tqdm

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
            df_eval = self.df.copy() #
            
        for k in ls_k:
            d_eval[task_name][lang][k] = df_eval.apply(lambda x: len(list((set(x["preds"][:k]) & set(x["gs"])))) > 0, axis=1).mean()
            # d_eval[task_name][lang]["individual"][k] = df_eval.explode("gs").apply(lambda x: x["gs"] in x["preds"][:k], axis=1).mean()

        # if output_folder is not None:
        #     eval_filename = "evaluation.json"
        #     output_file = os.path.join(output_folder, eval_filename)
        #     json.dump(d_eval, open(output_file, "w"), indent=4)
            
        return d_eval

class EmbeddingModel(BaseModel):
    def __init__(self, model_name, df_fc, device="cuda", show_progress_bar=True, batch_size=128, normalize_embeddings=True, k=10, model_type=None, **kwargs):
        super().__init__(device, show_progress_bar, batch_size, k)
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.normalize_embeddings = normalize_embeddings
        self.model_type = model_type
        
        if self.model_type == "jina":
            self.emb_fc = self.encode(df_fc["full_text"].values, task="retrieval.passage")
        else:
            self.emb_fc = self.encode(df_fc["full_text"].values)
            
        self.pos_to_idx = {pos: idx for pos, idx in enumerate(df_fc.index)}


    def encode(self, texts, task=None):
        
        if task is not None:
            return torch.tensor(self.model.encode(texts, show_progress_bar=self.show_progress_bar, 
                                              batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings,
                                              task=task)) #  device=self.device,
        else:
            return torch.tensor(self.model.encode(texts, show_progress_bar=self.show_progress_bar, 
                                                batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings))
        
    def train(self, texts):
        pass
    
    def similarity(self, emb1, emb2):
        return torch.mm(emb1, emb2.T).cpu().numpy()
    
    def predict(self, texts, scores=False, limit_k=True):
        
        if self.model_type == "jina":
            arr1 = self.encode(texts, task="retrieval.query")
        else:
            arr1 = self.encode(texts)
            
        sim = self.similarity(arr1, self.emb_fc)
        idx_sim = np.argsort(-sim, axis=1)
        # Apply the function element-wise to the array
        vectorized_map = np.vectorize(lambda x: self.pos_to_idx.get(x, None))
        
        if scores:
            if limit_k:
                return vectorized_map(idx_sim)[:, :self.k], np.sort(sim, axis=1)[:, ::-1][:, :self.k]
            else:
                return vectorized_map(idx_sim), sim
        else:
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
    
    def predict(self, posts, cands_list):
        # df_posts_dev["preds"] = df_posts_dev.apply(lambda x:cross_model.predict(x["full_text"], x["preds"]), axis=1)
        cands_list_text = [self.idx_to_text[pos] for pos in cands_list]
        pos_ids = self.model.rank(posts, cands_list_text, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, top_k=self.k, convert_to_numpy=True)
        pos_ids = [pos_id["corpus_id"] for pos_id in pos_ids]
        return np.array(cands_list)[pos_ids] # type: ignore

class IEModel(BaseModel):
    '''
    This model receives:
        - Fact-check claims dataset with a column called full_text
        - Spacy model name
    
    Applies Spacy NER to the full_text column and stores the entities in a set.
    Applies lemmatisation to the full_text column and stores the lemmas in a set.
    
    Compares the entities and lemmas of the fact-check claims with the entities and lemmas of the posts.
    Take the k first fact check for each post with the highest number of common entities and lemmas.
    '''
    
    def __init__(self, model_name, df_fc, device="cuda",  k=10, **kwargs):
        # "en_core_web_sm"
        if device == "cuda":
            spacy.prefer_gpu()
            
        self.model = spacy.load(model_name)
        # self.idx_to_text = df_fc["full_text"].to_dict()
        self.pos_to_idx = dict(enumerate(df_fc.index))
        self.vectorized_map = np.vectorize(lambda x: self.pos_to_idx.get(x, None))     
        self.emb_fc = self.encode(df_fc["full_text"].values)
           
        super().__init__(device, k=k)
    
    def encode(self, texts):
        processed = [
            {y.lemma_ for y in self.model(text) if not (y.is_stop or y.is_punct or y.is_space) and y.is_alpha}
            | set(self.extract_entities(text, self.model))
            for text in texts
        ]
        return np.array(processed)

    def train(self, texts):
        pass

    def predict(self, texts):
        emb_texts = self.encode(texts)#.values  # Convert to a list or numpy array for faster iteration
        fc_texts = self.emb_fc#.values

        # Calculate intersections efficiently
        preds = []
        for emb in emb_texts:
            common_counts = np.array([len(emb & fc_el) for fc_el in fc_texts])
            top_k_indices = np.argsort(-common_counts)[:self.k]  # Sort and pick top k indices
            preds.append([self.pos_to_idx[i] for i in top_k_indices])
        
        return np.array(preds)

    @staticmethod
    def get_word_intersection(x, y):
        set_words_in_common = set(x).intersection(y)
        return list(set_words_in_common)
    
    @staticmethod
    def extract_entities(text, model):
        # get hashtags
        hashtags = [word[1:] for word in text.split() if word.startswith("#")]
        text = text.replace("#", "")
        
        ents_0 = model(text).ents
        # get ents ngrams from 1 to 3

        return [ent.text for ent in ents_0] + hashtags

    # def predict(self, ):
        
