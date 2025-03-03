import ast
import json
from abc import abstractmethod
import numpy as np
from typing import List
import pandas as pd
import emoji
from tqdm import tqdm
import re
tqdm.pandas()

from src import cleaning
from src import config
class Dataset:
    """
    This is the base class for all datasets. It loads the data and tasks dictionary.
    Inputs:

    path: Path to the dataset
    tasks_path: Path to the tasks dictionary
    task_name: Name of the task (monolingual or crosslingual)
    lang: Language of the dataset (default: eng)
    version: Version of the dataset (default: None) Options ["english", "original"]
    index_col: Index column for the dataset (default: None) Important for splitting the dataset
    iter_cols: Columns that need to be iterated over (default: []) Important for preprocessing the dataset
    """
    def __init__(self, path: str, tasks_path: str, task_name:str, 
                 lang: str="eng", version:str="original", index_col:str=None, # type: ignore
                 iter_cols:List[str]=["text", "ocr", "instances"]):
        
        assert task_name in ["monolingual", "crosslingual"]
        assert version in ["english", "original"]
        
        self.path = path
        self.task_name = task_name
        self.tasks_path = tasks_path
        self.lang = lang
        self.index_col = index_col
        self.iter_cols = iter_cols
        self.version = version
        self.idx_lang = 0 if self.version == "original" else 1

        # Load tasks dictionary
        self.tasks = self.load_tasks()
        self.langs = list(self.tasks["monolingual"].keys())
        

        d_lan = {}
        # This ensures we don't get an error in multilingual case
        if self.lang != "all":
            d_lan = self.get_if_exists(self.tasks[self.task_name], self.lang)
        else:
            for lang in self.langs:
                d_lan_i = self.get_if_exists(self.tasks[self.task_name], lang)
                d_lan["posts_train"] = d_lan.get("posts_train", []) + d_lan_i.get("posts_train", [])
                d_lan["posts_dev"] = d_lan.get("posts_dev", []) + d_lan_i.get("posts_dev", [])
                d_lan["fact_checks"] = d_lan.get("fact_checks", []) + d_lan_i.get("fact_checks", [])

        
        self.idx_train, self.idx_dev = d_lan["posts_train"], d_lan["posts_dev"]
        self.idx_fc = d_lan["fact_checks"]

        # Be careful, it doesn't filter by language
        self.df = self.preprocess_data()

    @staticmethod
    def get_if_exists(dict, key):
        return dict[key] if key in dict else dict
    
    def load_data(self, indices=None):
        if self.index_col is None:
            # df = pd.read_csv("your_dataset.csv", sep=",", quoting=3, encoding="utf-8", engine="python")

            df = pd.read_csv(self.path, sep=",")
        else:
            df = pd.read_csv(self.path, sep=",").set_index(self.index_col)
        
        if indices is not None:
            df = df.loc[indices, :]
    
        import ast
        
        parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n').replace("(nan,", "(None,")) if s else s
        for col in self.iter_cols:
            df[col] = df[col].fillna("''").apply(parse_col)

        return df
    
    def __len__(self):
        return len(self.df)

    def load_tasks(self):
        return json.load(open(self.tasks_path))

    @abstractmethod
    def preprocess_data(self):
        df_proc = self.load_data()
        return df_proc
    
    def __repr__(self):
        if self.task_name == "monolingual":
            return f"Dataset -- {self.path}, Task: {self.task_name}, Lang: {self.lang}"
        else:
            return f"Dataset -- {self.path}, Task: {self.task_name}" 
    

class BasePostsDataset(Dataset):
    """
    This class is used to load the posts dataset.
    
    Attributes:
    - df: All the posts in every language.
    - df_train: Train posts in lang.
    - df_dev: Dev posts in lang.

    Inputs:

    posts_path: Path to the posts
    tasks_path: Path to the tasks dictionary
    task_name: Name of the task (monolingual or crosslingual)
    lang: Language of the dataset (default: eng)
    version: Version of the dataset (default: None) Options ["english", "original"]
    """
    iter_cols = ['instances', 'ocr', 'verdicts', 'text']
    index_col = "post_id"

    def __init__(self, posts_path, tasks_path, task_name, lang="eng", version="original", gs_path=None, **kwargs):
        super().__init__(posts_path, tasks_path, task_name, lang, index_col=self.index_col, iter_cols=self.iter_cols, version=version, **kwargs)

        self.gs_path = gs_path
        if gs_path:
            self.load_gs()

        self.df_train, self.df_dev = self.get_train_dev(self.df)


    def preprocess_data(self):

        df_posts = self.load_data(indices=self.idx_train+self.idx_dev)
        
        # Get the language of the text in order to stratify the data in crosslingual case
        text_lan = df_posts["text"].apply(lambda x: x[-1][0][0] if isinstance(x, tuple) else x)
        ocr_lan = df_posts["ocr"].apply(lambda x: x[0][-1][0][0] if isinstance(x, list) and len(x) > 0 and x is not None else "")
        
        # text_lan_simp = text_lan.apply(lambda x: x if (x in self.langs or len(x)==0) else "other")
        # ocr_lan_simp = ocr_lan.apply(lambda x: x if (x in self.langs or len(x)==0) else "other")
        
        df_posts["lan"] = text_lan
        df_posts.loc[text_lan.apply(len) == 0, "lan"] = ocr_lan.loc[text_lan.apply(len) == 0]
        
        df_posts["text"] = df_posts["text"].apply(lambda x: x[self.idx_lang] if isinstance(x, tuple) else x)
        df_posts["ocr"] = df_posts["ocr"].apply(lambda x: " ".join(trip[self.idx_lang] for trip in x) if x is not None else "")
        df_posts["verdicts"] = df_posts["verdicts"].apply(lambda x: x[0] if (isinstance(x, list))&(len(x)>0) else "")
        df_posts["instances"] = df_posts["instances"].apply(lambda x: [social for _, social in x] if len(x)>0 else [])
        df_posts["fb"] = df_posts["instances"].apply(lambda x: np.sum(np.array(x)=="fb"))
        df_posts["tw"] = df_posts["instances"].apply(lambda x: np.sum(np.array(x)=="tw"))
        df_posts["ig"] = df_posts["instances"].apply(lambda x: np.sum(np.array(x)=="ig"))
            
        df_posts.drop(columns=["instances"], inplace=True)
        return df_posts
    
    def load_gs(self):
        # group by post_id and get a list of all the fact-checks
        gs_col = pd.read_csv(str(self.gs_path)).groupby(self.index_col)["fact_check_id"].apply(list).reset_index()
        gs_col.columns = [self.index_col, "gs"]
        self.df = self.df.reset_index().merge(gs_col, on=self.index_col, how="left").fillna("")
        self.df["gs"] = self.df["gs"].map(lambda x: x if len(x) > 0 else [])
        self.df.set_index(self.index_col, inplace=True)

    def get_train_dev(self, df):
        if not self.index_col:
            raise ValueError("Index column for split not set (index_col)")
        return df.loc[self.idx_train, :], df.loc[self.idx_dev, :]#.drop(columns=["gs"])
    
    def __repr__(self):
        return super().__repr__().replace("Dataset", "BasePostsDataset") + f", Train: {self.df_train.shape}, Dev: {self.df_dev.shape}"
    

class BaseFactCheckDataset(Dataset):

    """
    This class is used to load the fact-check dataset.

    Inputs:

    fact_check_path: Path to the fact-checks
    tasks_path: Path to the tasks dictionary
    task_name: Name of the task (monolingual or crosslingual)
    lang: Language of the dataset (default: eng)
    version: Version of the dataset (default: None) Options ["english", "original"]
    """

    iter_cols = ['title', 'claim', "instances"]
    index_col = "fact_check_id"

    def __init__(self, fact_check_path, tasks_path, task_name, lang="eng", version="original", **kwargs):
        super().__init__(fact_check_path, tasks_path, task_name, lang, index_col=self.index_col, iter_cols=self.iter_cols, version=version, **kwargs)
        self.df = self.preprocess_data()
        # self.df = self.df.loc[self.idx_fc, :]

    def preprocess_data(self):
        df_fact_check = self.load_data(indices=self.idx_fc)
        text_lan = df_fact_check["claim"].apply(lambda x: x[-1][0][0] if isinstance(x, tuple) else x)
        df_fact_check["lan"] = text_lan
        df_fact_check["claim"] = df_fact_check["claim"].apply(lambda x: x[self.idx_lang] if isinstance(x, tuple) else x)
        df_fact_check["title"] = df_fact_check["title"].apply(lambda x: x[self.idx_lang] if isinstance(x, tuple) else x)
        df_fact_check["instances"] = df_fact_check["instances"].apply(lambda x: [url for _, url in x] if len(x)>0 else [])
        

        
        return df_fact_check
    
    def __repr__(self):
        return super().__repr__().replace("Dataset", "BaseFactCheckDataset") + f", Fact Checks: {self.df.shape}"
    

class TextConcatPosts(BasePostsDataset):
    """
    This class is used to load the fact-check dataset.

    Inputs:

    fact_check_path: Path to the fact-checks
    tasks_path: Path to the tasks dictionary
    task_name: Name of the task (monolingual or crosslingual)
    lang: Language of the dataset (default: eng)
    version: Version of the dataset (default: None) Options ["english", "original"]
    """
    
    def __init__(self, posts_path, tasks_path, task_name, lang="eng", version="original", gs_path=None, demojize=False, prefix="", cleaning_function=None, clean=False, lang_prompt=False, **kwargs):
        self.demojize = demojize
        self.prefix = prefix
        self.clean = clean
        self.cleaning_function = cleaning_function
        self.lang_prompt = lang_prompt
        super().__init__(posts_path, tasks_path, task_name, lang, version, gs_path, **kwargs)
    

    def preprocess_data(self):
        df_posts = super().preprocess_data()
        df_posts["full_text"] = self.prefix + df_posts["ocr"] + " " + df_posts["text"]
        
        if self.lang_prompt:
            df_lans = df_posts["lan"].map(config.LANG_COUNTRY).fillna("")
            df_posts["full_text"] = df_lans + " : " + df_posts["full_text"]
        
        if self.demojize:
            df_posts["full_text"] = df_posts["full_text"].apply(lambda x: emoji.demojize(x))
        
        if self.clean:
            df_posts["full_text"] = df_posts["full_text"].str.replace(cleaning.url_regex, "", regex=True)\
                                                         .str.replace(cleaning.emoji_regex, "", regex=True)\
                                                         .str.replace(cleaning.sentence_stop_regex, ".", regex=True)\
                                                         .str.replace("\[.+\]", "", regex=True)
        if self.cleaning_function:
            df_posts["full_text"] = self.cleaning_function(df_posts["full_text"].values)
            
        return df_posts
    
class TextConcatFactCheck(BaseFactCheckDataset):
    """
    This class is used to load the fact-check dataset.

    Inputs:

    fact_check_path: Path to the fact-checks
    tasks_path: Path to the tasks dictionary
    task_name: Name of the task (monolingual or crosslingual)
    lang: Language of the dataset (default: eng)
    version: Version of the dataset (default: None) Options ["english", "original"]
    """
    
    def __init__(self, fact_check_path, tasks_path, task_name, lang="eng", version="original", demojize=False, prefix="", clean=False, cleaning_function=None, **kwargs):
        self.demojize = demojize
        self.prefix = prefix
        self.clean = clean
        self.cleaning_function = cleaning_function
        super().__init__(fact_check_path, tasks_path, task_name, lang, version, **kwargs)

    def preprocess_data(self):
        df_fact_check = super().preprocess_data()
        df_fact_check["full_text"] = self.prefix + df_fact_check["title"] + " " + df_fact_check["claim"]
        # df_fact_check["full_text"] = df_fact_check["full_text"].str.lower()
        if self.demojize:
            df_fact_check["full_text"] = df_fact_check["full_text"].apply(lambda x: emoji.demojize(x))
        
        if self.clean:
            df_fact_check["full_text"] = df_fact_check["full_text"].str.replace(cleaning.url_regex, "", regex=True)\
                                                                    .str.replace(cleaning.emoji_regex, "", regex=True)\
                                                                    .str.replace(cleaning.sentence_stop_regex, ".", regex=True)

        if self.cleaning_function is not None and "df_clean" not in self.__dict__:
            self.df_clean = df_fact_check.copy()
            self.df_clean["full_text"] = self.cleaning_function(self.df_clean["full_text"].values)
            
        return df_fact_check    
    
