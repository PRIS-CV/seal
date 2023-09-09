from collections import Counter
import numpy as np
import math
import torch
from typing import Union

from . import metric
from .metric import Metric


def entropy(probs: list):
    r"""Calculate the information entropy of the discrete probability distribution.
    """
    probs = np.array(probs)
    return -np.sum(probs * np.log2(probs))


@metric("GroupDiversity")
class GroupDiversity(Metric):
    
    def __init__(self, name, **kwargs) -> None:
        super().__init__(name, **kwargs)
    
    def calculate_metric(self, group_retrieval_id):
        group_probs = {}
        group_diversity = {}
        for g in group_retrieval_id:
            group_size = len(group_retrieval_id[g])
            group_probs[g] = {g_id: g_freq / group_size for g_id, g_freq in Counter(group_retrieval_id[g]).items()}
            group_diversity[g] = entropy(list(group_probs[g].values())) / math.log2(group_size)
        self._result.update(group_diversity)
        self._result.update({"mean_diversity": sum(list(group_diversity.values())) / len(list(group_diversity.keys()))})
    
    def reset(self):
        self._result = {}
        
    