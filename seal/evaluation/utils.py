import numpy as np
import torch
from torch import Tensor

def load_to_device(data, device:torch.device):
    if isinstance(data, Tensor):
        data = data.to(device)
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = load_to_device(v, device)
    elif isinstance(data, list):
        data = [load_to_device(item, device) for item in data]
    else:
        raise TypeError(f"Cannot load type {type(data)} to device {device}")
    
    return data

def top_K_indexs(array, K=15):
    return np.argpartition(array, -K, axis=-1)[-K:]

def top_K_values(array, K=15):
    """Keeps only topK largest values in array. other will be 0
    """
    indexes = np.argpartition(array, -K, axis=-1)[-K:]
    A = set(indexes)
    B = set(list(range(array.shape[0])))
    B -= A
    array[list(B)]=0
    return array
