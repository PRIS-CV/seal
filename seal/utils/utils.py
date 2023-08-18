import random
import numpy as np
import os.path as op
import torch
from torch import Tensor
from yacs.config import CfgNode




def set_seed(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

def convert_cfg_to_dict(root_node):
    r"""Convert CfgNode to a dict.
    """

    def convert_to_dict(cfg_node, key_list):
        if not isinstance(cfg_node, CfgNode):
            return cfg_node
        else:
            cfg_dict = dict(cfg_node)
            for k, v in cfg_dict.items():
                cfg_dict[k] = convert_to_dict(v, key_list + [k])
            return cfg_dict

    self_as_dict = convert_to_dict(root_node, [])
    return self_as_dict


def get_constr_out(x, R):
    r"""
        Given the output of the neural network x, this function 
        returns the output of MCM given the hierarchy constraint 
        expressed in the adjacency matrix R. 
    """

    
    
    c_out = x
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    final_out, max_index = torch.max(R_batch * c_out.double(), dim = 2)
    return final_out, max_index