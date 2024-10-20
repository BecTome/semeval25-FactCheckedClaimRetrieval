from sentence_transformers import SentenceTransformer
import torch
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name, df_fc, device="cuda", show_progress_bar=True, batch_size=128, normalize_embeddings=True, k=10):
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.emb_fc = self.encode(df_fc["full_text"].values)
        self.pos_to_idx = {pos: idx for pos, idx in enumerate(df_fc.index)}
        self.k = k

    def encode(self, texts):
        return torch.tensor(self.model.encode(texts, device=self.device, show_progress_bar=self.show_progress_bar, 
                                              batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings))
    
    def similarity(self, emb1, emb2):
        return torch.mm(emb1, emb2.T).cpu().numpy()
    
    def predict(self, texts):
        arr1 = self.encode(texts)
        sim = self.similarity(arr1, self.emb_fc)
        idx_sim = np.argsort(sim, axis=1)[:, ::-1][:, :self.k]
        # Apply the function element-wise to the array
        vectorized_map = np.vectorize(lambda x: self.pos_to_idx.get(x, None))
        return vectorized_map(idx_sim)
