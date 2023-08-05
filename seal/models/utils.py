from collections import OrderedDict
import numpy as np
import torch
from torch import nn, Tensor
from torchvision.ops import roi_align

from seal.data.utils import load_json


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict


def add_weight_decay_and_lr(model, weight_decay=1e-4, fast_lr=0., base_lr=0., skip_list=(), show_detail=False):

    fast_params = model.fast_params()

    decay = []
    decay_name = []
    
    decay_fast = []
    decay_fast_name = []
    
    no_decay = []
    no_decay_name = []
    
    no_decay_fast = []
    no_decay_fast_name = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            if name in fast_params:
                no_decay_fast.append(param)
                no_decay_fast_name.append(name)
            else:
                no_decay.append(param)
                no_decay_name.append(name)
        else:
            if name in fast_params:
                decay_fast.append(param)
                decay_fast_name.append(name)
            else:
                decay.append(param)
                decay_name.append(name)
                
    if show_detail:
        print("\nSet The Weight Decay And Lr ...")
        print(f"\nWeight Decay: {0.} Learning Rate: {base_lr} Parameters: {no_decay_name}\n")
        print(f"Weight Decay: {0.} Learning Rate: {fast_lr} Parameters: {no_decay_fast_name}\n")
        print(f"Weight Decay: {weight_decay} Learning Rate: {base_lr} Parameters: {decay_name}\n")
        print(f"Weight Decay: {weight_decay} Learning Rate: {fast_lr} Parameters: {decay_fast_name}\n")
    
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': no_decay_fast, 'weight_decay': 0., 'lr':fast_lr},
        {'params': decay, 'weight_decay': weight_decay},
        {'params': decay_fast, 'weight_decay': weight_decay, 'lr':fast_lr}
    ]


def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out


def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out


class ROIAlign(nn.Module):
    
    def __init__(self, spatial_scale, output_size, aligned:bool=False) -> None:
        super().__init__()
        
        self.spatial_scale = spatial_scale
        self.output_size = output_size
        self.aligned = aligned

    def forward(self, feature, bboxes):
        
        return roi_align(
            input=feature, 
            boxes=bboxes, 
            output_size=self.output_size, 
            spatial_scale=self.spatial_scale, 
            sampling_ratio=2,
            aligned=self.aligned
        )


class Completer:

    def __init__(self, cfg) -> None:
        
        self.num_classes = cfg.num_classes

        self.f_pos_relations = cfg.f_pos_relations
        self.f_neg_relations = cfg.f_neg_relations

        self.relation_threshold = cfg.relation_threshold
        self.likelihood_topk = cfg.likelihood_topk
        self.freq_threshold = cfg.freq_threshold

        self.f_obj2idx = cfg.f_obj2idx
        self.f_obj_freq = cfg.f_obj_freq
        self.f_query = cfg.f_query

        self.positive = True
        self.negative = False

        self.num_levels = len(self.num_classes)
        self.obj_freq = self.load_obj_freq(cfg.f_obj_freq)

        self.pos_mats, self.neg_mats = [], []
        print("Loading Relation Matrix ...")
        for i in range(self.num_levels):
            pos_mat, neg_mat = self.load_relation(cfg.f_pos_relations[i], cfg.f_neg_relations[i])
            if self.positive:
                self.pos_mats.append(self.combine_global_and_local_mat(pos_mat))
            if self.negative:
                self.neg_mats.append(self.combine_global_and_local_mat(neg_mat))
        print("Finish ...")

    def load_obj_freq(self, f_obj_freq):
        obj_freq = torch.tensor(list(load_json(f_obj_freq)["object_frequence"].values()))
        return obj_freq

    def load_relation(self, f_pos_relation, f_neg_relation):
        pos_relation, neg_relation = None, None
        if self.positive:
            pos_relation = np.load(f_pos_relation)
        if self.negative:
            neg_relation = np.load(f_neg_relation)
        return pos_relation, neg_relation

    def combine_global_and_local_mat(self, adj_mat, beta=0.8):
        if isinstance(adj_mat, np.ndarray):
            adj_mat = torch.from_numpy(adj_mat)
        
        o_mat = adj_mat[:-1]
        g_mat = adj_mat[-1].unsqueeze(0)

        mask = self.obj_freq < self.freq_threshold
        o_mat[mask] = beta * g_mat + (1 - beta) * o_mat[mask]

        o_mat = (o_mat > self.relation_threshold).float()

        return o_mat

    def complete_targets(self, xs_neg, level, obj_index, targets=None, targets_weights=None):
        pos_mat = self.pos_mats[level][obj_index]
        top_k_mask = self.get_topk_likelihood_index(targets, xs_neg, self.likelihood_topk)
        if targets is not None:
            neighbors_mask = self.get_neighbor_index(targets, pos_mat)
            union_mask, pp_mask, ig_mask = self.get_intersection_and_difference(top_k_mask, neighbors_mask)
        else:
            union_mask = top_k_mask
        
        if targets is not None:
            targets[pp_mask] = 1
            targets_weights[ig_mask] = 0

        return targets, targets_weights, union_mask

    def __call__(self, logits, obj_index, level, targets=None):
        with torch.no_grad():
            xs_pos = torch.sigmoid(logits)
            xs_neg = 1.0 - xs_pos
            targets_weights = None
            if targets is not None:
                targets_weights = torch.ones_like(xs_neg)
            new_targets, targets_weights, mask = self.complete_targets(xs_neg, level, obj_index, targets, targets_weights)
        return new_targets, targets_weights, mask

    def get_topk_likelihood_index(self, targets: Tensor, xs_neg_prob: Tensor, num_top_k: int):
        with torch.no_grad():
            top_prob_index = torch.zeros_like(xs_neg_prob).bool()                                      
            rank_index = torch.argsort(xs_neg_prob, dim=-1)       
            top_prob_index[:, rank_index[:num_top_k]] = True
            if targets is not None:
                unlable_index = (targets == 2).float()
                return torch.logical_and(unlable_index, top_prob_index)
            else:
                return top_prob_index
        

    def get_neighbor_index(self, targets: Tensor, adj_mat: Tensor):
        r"""
            Args:
                targets (Tensor): 
                    The ground truth label whose shape is [B, C].
                adj_mat (Tensor):
                    The adjacent matrix whose shape is [B, C, C].
                return (Tensor):
                    The neighbors bool tensor whose shape is [B, C]
        """
        with torch.no_grad():
            adj_mat = adj_mat.to(targets.device)
            anno_pos_index = (targets == 1).unsqueeze(1).float()
            neibors = torch.bmm(anno_pos_index, adj_mat).squeeze(1)
            return neibors == 1

    def get_intersection_and_difference(self, A, B):
        r"""Return the intersection and difference bool mask of mask A and mask B 
            Args:
                A (Tensor): The bool tensor.
                B (Tensor): The bool tensor.
            Return:
                Tensor: The intersection mask (A \intersection B)
                Tensor: The difference mask (A \union B - (A \intersection B))
        """
        union = torch.logical_or(A, B)
        intersection = torch.logical_and(A, B)
        difference = torch.logical_xor(union, intersection)
        return union, intersection, difference
    