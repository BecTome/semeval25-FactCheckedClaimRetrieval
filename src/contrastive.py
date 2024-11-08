import numpy as np
import pandas as pd

def generate_triplets(df_queries, df_passages, teacher_model, n_candidates, neg_perc_threshold, random_state=42):
        # Retrieve similarity scores
        df_train_posts_pairs = df_queries[["full_text", "gs"]].copy()

        idx, sim = teacher_model.predict(df_train_posts_pairs["full_text"].values, scores=True, limit_k=False)
        sorted_sim = np.sort(sim, axis=1)[:, ::-1]

        # n_candidates = 4
        # neg_perc_threshold = 0.9
        
        df_train_posts_pairs["post_pos"] = np.arange(len(df_train_posts_pairs))
        df_train_posts_pairs = df_train_posts_pairs.explode("gs", ignore_index=True)
        
        d_fc_idx_text = df_passages["full_text"].to_dict()
        df_train_posts_pairs["gs_text"] = df_train_posts_pairs["gs"].map(d_fc_idx_text)
        
        d_idx_to_pos = {idx: pos for pos, idx in teacher_model.pos_to_idx.items()}
        df_train_posts_pairs["gs_pos"] = df_train_posts_pairs["gs"].map(d_idx_to_pos)
        df_train_posts_pairs["gs_score"] = df_train_posts_pairs.apply(lambda x: sim[x["post_pos"], x["gs_pos"]], axis=1)
        df_train_posts_pairs["neg_thres"] = df_train_posts_pairs["gs_score"] * neg_perc_threshold
        # df_train_posts_pairs["naive_topk"] = df_train_posts_pairs.apply(lambda x: idx[x["post_pos"]][idx[x["post_pos"]] != x["gs_pos"]][:n_candidates], axis=1)

        mask_neg_thres = lambda col1, col2: sorted_sim[col1] < col2
        mask_exclude_gs = lambda col1, col2: idx[col1] != col2

        df_train_posts_pairs["idx_neg_thres"] = df_train_posts_pairs.apply(lambda x: idx[x["post_pos"]][(mask_neg_thres(x["post_pos"], x["neg_thres"])&\
                                                                                                        mask_exclude_gs(x["post_pos"], x["gs"]))][:n_candidates], axis=1)

        df_negatives = df_train_posts_pairs[["full_text", "idx_neg_thres"]].copy()
        df_negatives = df_negatives.explode("idx_neg_thres", ignore_index=True)
        df_negatives["neg_text"] = df_negatives["idx_neg_thres"].map(d_fc_idx_text)
        df_negatives["label"] = 0
        df_negatives.drop(columns="idx_neg_thres", inplace=True)

        df_positives = df_train_posts_pairs[["full_text", "gs_text"]].copy()
        df_positives["label"] = 1
        
        df_negatives.rename(columns={"full_text": "query", "neg_text": "passage"}, inplace=True)
        df_positives.rename(columns={"full_text": "query", "gs_text": "passage"}, inplace=True)

        df_cl = pd.concat([df_positives, df_negatives], ignore_index=True).sample(frac=1., random_state=random_state)
        
        df_cl = df_cl.dropna()

        return df_cl

def generate_triplets_candidates(df_queries, df_passages, teacher_model, n_candidates, neg_perc_threshold, random_state=42):
        # Retrieve similarity scores
        df_train_posts_pairs = df_queries[["full_text", "gs", "candidates"]].copy()
        
        idx, sim = teacher_model.predict(df_train_posts_pairs["full_text"].values, scores=True, limit_k=False)
        sorted_sim = np.sort(sim, axis=1)[:, ::-1]

        # n_candidates = 4
        # neg_perc_threshold = 0.9
        
        df_train_posts_pairs["post_pos"] = np.arange(len(df_train_posts_pairs))
        df_train_posts_pairs = df_train_posts_pairs.explode("gs", ignore_index=True)
        
        d_fc_idx_text = df_passages["full_text"].to_dict()
        df_train_posts_pairs["gs_text"] = df_train_posts_pairs["gs"].map(d_fc_idx_text)
        
        d_idx_to_pos = {idx: pos for pos, idx in teacher_model.pos_to_idx.items()}
        df_train_posts_pairs["gs_pos"] = df_train_posts_pairs["gs"].map(d_idx_to_pos)
        # df_train_posts_pairs["cand_pos"] = df_train_posts_pairs["candidates"].map(d_idx_to_pos)
        df_train_posts_pairs["gs_score"] = df_train_posts_pairs.apply(lambda x: sim[x["post_pos"], x["gs_pos"]], axis=1)
        df_train_posts_pairs["neg_thres"] = df_train_posts_pairs["gs_score"] * neg_perc_threshold
        # df_train_posts_pairs["naive_topk"] = df_train_posts_pairs.apply(lambda x: idx[x["post_pos"]][idx[x["post_pos"]] != x["gs_pos"]][:n_candidates], axis=1)

        mask_neg_thres = lambda col1, col2: sorted_sim[col1] < col2
        mask_exclude_gs = lambda col1, col2: idx[col1] != col2
        mask_in_candidates = lambda col1, ls: len(np.intersect1d(idx[col1], ls)) > 0
        
        df_train_posts_pairs["idx_neg_candidates"] = df_train_posts_pairs.apply(lambda x: idx[x["post_pos"]][(mask_neg_thres(x["post_pos"], x["neg_thres"])&\
                                                                                                                        mask_exclude_gs(x["post_pos"], x["gs"])&\
                                                                                                                        mask_in_candidates(x["post_pos"], x["candidates"]))], 
                                                                                axis=1)

        df_train_posts_pairs["idx_neg_candidates"] = df_train_posts_pairs["idx_neg_candidates"].apply(lambda x: x if len(x) < n_candidates else np.random.choice(x, n_candidates, replace=False))
        
        df_negatives = df_train_posts_pairs[["post_pos", "full_text", "idx_neg_candidates"]].copy()
        df_negatives = df_negatives[df_negatives["idx_neg_candidates"].apply(len) > 0]
        df_negatives = df_negatives.explode("idx_neg_candidates", ignore_index=True)
        df_negatives["neg_text"] = df_negatives["idx_neg_candidates"].map(d_fc_idx_text)
        # use tanh to normalize the scores
        soft_function = np.tanh
        df_negatives["label"] = df_negatives.apply(lambda x: soft_function(sim[x["post_pos"], d_idx_to_pos[x["idx_neg_candidates"]]]), axis=1)
        df_negatives.drop(columns=["post_pos", "idx_neg_candidates"], inplace=True)

        df_positives = df_train_posts_pairs[["full_text", "gs_text"]].copy()
        df_positives["label"] = 1.
        
        df_negatives.rename(columns={"full_text": "query", "neg_text": "passage"}, inplace=True)
        df_positives.rename(columns={"full_text": "query", "gs_text": "passage"}, inplace=True)

        df_cl = pd.concat([df_positives, df_negatives], ignore_index=True).sample(frac=1., random_state=random_state)
        
        df_cl = df_cl.dropna()

        return df_cl
        
def generate_triplets_2ofeach(df_queries, df_passages, teacher_model, n_candidates, neg_perc_threshold, random_state=42):
       # Retrieve similarity scores
        df_train_posts_pairs = df_queries[["full_text", "gs"]].copy()
        df_train_posts_pairs["gs_count"] = df_train_posts_pairs["gs"].apply(len)
        
        idx, sim = teacher_model.predict(df_train_posts_pairs["full_text"].values, scores=True, limit_k=False)
        sorted_sim = np.sort(sim, axis=1)[:, ::-1]

        # n_candidates = 4
        # neg_perc_threshold = 0.9
        
        df_train_posts_pairs["post_pos"] = np.arange(len(df_train_posts_pairs))
        df_train_posts_pairs = df_train_posts_pairs.explode("gs", ignore_index=True)
        
        d_fc_idx_text = df_passages["full_text"].to_dict()
        df_train_posts_pairs["gs_text"] = df_train_posts_pairs["gs"].map(d_fc_idx_text)
        
        d_idx_to_pos = {idx: pos for pos, idx in teacher_model.pos_to_idx.items()}
        df_train_posts_pairs["gs_pos"] = df_train_posts_pairs["gs"].map(d_idx_to_pos)
        # df_train_posts_pairs["cand_pos"] = df_train_posts_pairs["candidates"].map(d_idx_to_pos)
        df_train_posts_pairs["gs_score"] = df_train_posts_pairs.apply(lambda x: sim[x["post_pos"], x["gs_pos"]], axis=1)
        df_train_posts_pairs["neg_thres"] = df_train_posts_pairs["gs_score"] * neg_perc_threshold
        # df_train_posts_pairs["naive_topk"] = df_train_posts_pairs.apply(lambda x: idx[x["post_pos"]][idx[x["post_pos"]] != x["gs_pos"]][:n_candidates], axis=1)

        mask_exclude_gs = lambda col1, col2: idx[col1] != col2
        mask_in_candidates = lambda col1, ls: len(np.intersect1d(idx[col1], ls)) > 0
        
        df_train_posts_pairs["idx_neg_candidates"] = df_train_posts_pairs.apply(lambda x: idx[x["post_pos"]][mask_exclude_gs(x["post_pos"], x["gs"])],axis=1)

        # take the two with highest similarity, two with lowest similarity, and two with middle similarity. Make sure the list has at least 6 elements
        def get2timesgs(ls):
                toretrieve = 2 * ls["gs_count"]
                if len(ls["idx_neg_candidates"]) <= toretrieve:
                        return ls["idx_neg_candidates"]
                if ls["gs_count"] == 1:
                        return ls["idx_neg_candidates"][0] + ls["idx_neg_candidates"][-1] + ls["idx_neg_candidates"][len(ls["idx_neg_candidates"])//2-1]
                else:
                        return ls["idx_neg_candidates"][:2] + ls["idx_neg_candidates"][-2:] + ls["idx_neg_candidates"][len(ls["idx_neg_candidates"])//2-1:len(ls["idx_neg_candidates"])//2+1]
        
        df_train_posts_pairs["idx_neg_candidates"] = df_train_posts_pairs.apply(get2timesgs, axis=1)
        
        df_negatives = df_train_posts_pairs[["post_pos", "full_text", "idx_neg_candidates"]].copy()
        df_negatives = df_negatives[df_negatives["idx_neg_candidates"].apply(len) > 0]
        df_negatives = df_negatives.explode("idx_neg_candidates", ignore_index=True)
        df_negatives["neg_text"] = df_negatives["idx_neg_candidates"].map(d_fc_idx_text)
        # use tanh to normalize the scores
        soft_function = np.tanh
        df_negatives["label"] = df_negatives.apply(lambda x: soft_function(sim[x["post_pos"], d_idx_to_pos[x["idx_neg_candidates"]]]), axis=1)
        df_negatives.drop(columns=["post_pos", "idx_neg_candidates"], inplace=True)

        df_positives = df_train_posts_pairs[["full_text", "gs_text"]].copy()
        df_positives["label"] = 1.
        
        df_negatives.rename(columns={"full_text": "query", "neg_text": "passage"}, inplace=True)
        df_positives.rename(columns={"full_text": "query", "gs_text": "passage"}, inplace=True)

        df_cl = pd.concat([df_positives, df_negatives], ignore_index=True).sample(frac=1., random_state=random_state)
        
        df_cl = df_cl.dropna()

        return df_cl