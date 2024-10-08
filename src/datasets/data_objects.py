from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

@dataclass
class Post:
    id: str
    text: str

    def __repr__(self):
        return f"Post(id={self.id}, text={self.text})"

@dataclass
class Fact:
    id: str
    claim: str
    verdict: Optional[str] = None

    def __repr__(self):
        return f"Fact(id={self.id}, claim={self.claim}, verdict={self.verdict})"
    
@dataclass
class FactCheck:
    post_id: str
    fact_ids: List[str]

    def __repr__(self):
        return f"FactCheck(id={self.id}, fact_ids={self.fact_ids})"