import ast
import json
from abc import abstractmethod
import numpy as np
from typing import List
import pandas as pd

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
                 lang: str="eng", version:str=None, index_col:str=None, 
                 iter_cols:List[str]=[]):
        
        assert task_name in ["monolingual", "crosslingual"]
        self.path = path
        self.task_name = task_name
        self.tasks_path = tasks_path
        self.lang = lang
        self.index_col = index_col
        self.iter_cols = iter_cols

        # Load tasks dictionary
        self.tasks = self.load_tasks()
        self.langs = list(self.tasks["monolingual"].keys())

        # This ensures we don't get an error in multilingual case
        d_lan = self.get_if_exists(self.tasks[self.task_name], self.lang)
        self.idx_train, self.idx_dev = d_lan["posts_train"], d_lan["posts_dev"]
        self.idx_fc = d_lan["fact_checks"]

        self.df = self.preprocess_data()

    @staticmethod
    def get_if_exists(dict, key):
        return dict[key] if key in dict else dict
    
    def load_data(self):
        if self.index_col is None:
            df = pd.read_csv(self.path).fillna('')
        else:
            df = pd.read_csv(self.path).fillna('').set_index(self.index_col)
        
        parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n')) if s else s
        for col in self.iter_cols:
            df[col] = df[col].apply(parse_col)

        return df

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

    Inputs:

    posts_path: Path to the posts
    tasks_path: Path to the tasks dictionary
    task_name: Name of the task (monolingual or crosslingual)
    lang: Language of the dataset (default: eng)
    version: Version of the dataset (default: None) Options ["english", "original"]
    """
    iter_cols = ['instances', 'ocr', 'verdicts', 'text']
    index_col = "post_id"

    def __init__(self, posts_path, tasks_path, task_name, lang="eng", version=None):
        super().__init__(posts_path, tasks_path, task_name, lang, index_col=self.index_col, iter_cols=self.iter_cols, version=version)

        self.df_train, self.df_dev = self.get_train_dev(self.df)

    def preprocess_data(self):
        df_posts = self.load_data()
        df_posts["text"] = df_posts["text"].apply(lambda x: x[0] if isinstance(x, tuple) else x)
        df_posts["ocr"] = df_posts["ocr"].apply(lambda x: " ".join(trip[0] for trip in x))
        df_posts["verdicts"] = df_posts["verdicts"].apply(lambda x: x[0] if (isinstance(x, list))&(len(x)>0) else "")
        df_posts["instances"] = df_posts["instances"].apply(lambda x: [social for _, social in x] if len(x)>0 else [])
        df_posts["fb"] = df_posts["instances"].apply(lambda x: np.sum(np.array(x)=="fb"))
        df_posts["tw"] = df_posts["instances"].apply(lambda x: np.sum(np.array(x)=="tw"))
        df_posts["ig"] = df_posts["instances"].apply(lambda x: np.sum(np.array(x)=="ig"))

        df_posts.drop(columns=["instances"], inplace=True)
        return df_posts
    
    def get_train_dev(self, df):
        if not self.index_col:
            raise ValueError("Index column for split not set (index_col)")
        return df.loc[self.idx_train, :], df.loc[self.idx_dev, :]
    
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

    def __init__(self, fact_check_path, tasks_path, task_name, lang="eng", version=None):
        super().__init__(fact_check_path, tasks_path, task_name, lang, index_col=self.index_col, iter_cols=self.iter_cols, version=version)
        self.df = self.preprocess_data()
        self.df = self.df.loc[self.idx_fc, :]

    def preprocess_data(self):
        df_fact_check = self.load_data()
        df_fact_check["claim"] = df_fact_check["claim"].apply(lambda x: x[0] if isinstance(x, tuple) else x)
        df_fact_check["title"] = df_fact_check["title"].apply(lambda x: x[0] if isinstance(x, tuple) else x)
        df_fact_check["instances"] = df_fact_check["instances"].apply(lambda x: [url for _, url in x] if len(x)>0 else [])
        return df_fact_check
    
    def __repr__(self):
        return super().__repr__().replace("Dataset", "BaseFactCheckDataset") + f", Fact Checks: {self.df.shape}"
    