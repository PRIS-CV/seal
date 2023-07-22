import numpy as np
import os.path as op
import torch
from torch import Tensor

from info.data.utils import load_json
from info.models.loss import loss
from info.models.loss.loss import Loss

@loss("MPP")
class MPP(Loss):

    def __init__(self, cfg, clip=0.05):

        super().__init__("MPP")
        
        assert op.exists(cfg.f_pos_relation),   "Please calculate the conditional probability first."
        assert op.exists(cfg.f_obj2idx),        "Please generate the `object_index.json` file first."
        assert op.exists(cfg.f_obj_freq),       "Please calculate the object frequence first."
        self.clip = clip
        self.gamma_pos = cfg.gamma_pos
        self.gamma_neg = cfg.gamma_neg
        self.gamma_unl = cfg.gamma_unl
        self.alpha_pos = cfg.alpha_pos
        self.alpha_neg = cfg.alpha_neg
        self.alpha_unl = cfg.alpha_unl
        self.positive = cfg.positive
        self.negative = cfg.negative
        self.f_pos_relation = cfg.f_pos_relation
        self.f_neg_relation = cfg.f_neg_relation
        self.relation_threshold = cfg.relation_threshold
        self.obj2idx = load_json(cfg.f_obj2idx)
        self.likelihood_topk = cfg.likelihood_topk
        self.f_obj_freq = cfg.f_obj_freq
        self.freq_threshold = cfg.freq_threshold

        print("Loading Attachment ...")
        self.obj_freq = self.load_obj_freq()
        pos_mat, neg_mat = self.load_relation()
        self.register_buffer("pos_mat", self.combine_global_and_local_mat(pos_mat))
        # self.register_buffer("neg_mat", self.combine_global_and_local_mat(neg_mat))
        print("Load Mat Finish")

    def combine_global_and_local_mat(self, adj_mat, beta=0.8):
        if isinstance(adj_mat, np.ndarray):
            adj_mat = torch.from_numpy(adj_mat)
        
        o_mat = adj_mat[:-1]
        g_mat = adj_mat[-1].unsqueeze(0)

        mask = self.obj_freq < self.freq_threshold
        o_mat[mask] = beta * g_mat + (1 - beta) * o_mat[mask]

        o_mat = (o_mat > self.relation_threshold).float()

        return o_mat


    def load_obj_freq(self):
        obj_freq = torch.tensor(list(load_json(self.f_obj_freq)["object_frequence"].values()))
        return obj_freq

    def load_relation(self):
        pos_relation, neg_relation = None, None
        if self.positive:
            pos_relation = np.load(self.f_pos_relation)
        if self.negative:
            neg_relation = np.load(self.f_neg_relation)
        return pos_relation, neg_relation

    def compelete_targets(self, xs_neg, targets, targets_weights, adj_mat):
        
        top_k_mask = get_topk_likelihood_index(targets, xs_neg, 5)
        neighbors_mask = get_neighbor_index(targets, adj_mat)
        union_mask, pp_mask, ig_mask = get_intersection_and_difference(top_k_mask, neighbors_mask)

        targets[pp_mask] = 1
        targets_weights[ig_mask] = 0

        return targets, targets_weights, union_mask

    def forward(self, logits, targets, obj_index):

        # Positive, Negative and Un-annotated indexes
        
        adj_mat = self.pos_mat[obj_index]
        
        device = logits.device

        # Activation
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)
        
        targets_weights = torch.ones_like(targets, dtype=torch.float32).to(device)
        targets, targets_weights, union_mask = self.compelete_targets(xs_neg, targets, targets_weights, adj_mat)

        targets_pos = (targets == 1).float()
        targets_neg = (targets == 0).float()
        targets_unl = (targets == 2).float()

        # Loss calculation
        BCE_pos = self.alpha_pos * targets_pos * torch.log(torch.clamp(xs_pos, min=1e-8))
        BCE_neg = self.alpha_neg * targets_neg * torch.log(torch.clamp(xs_neg, min=1e-8))
        BCE_unl = self.alpha_unl * targets_unl * torch.log(torch.clamp(xs_neg, min=1e-8))

        BCE_loss = BCE_pos + BCE_neg + BCE_unl

        # Adding asymmetric gamma weights
        with torch.no_grad():
            asymmetric_w = torch.pow(1 - xs_pos * targets_pos - xs_neg * (targets_neg + targets_unl),
                                     self.gamma_pos * targets_pos + self.gamma_neg * targets_neg +
                                     self.gamma_unl * targets_unl)
        BCE_loss *= asymmetric_w

        # partial labels weights
        BCE_loss *= targets_weights

        return -BCE_loss.sum()
    

def get_topk_likelihood_index(targets: Tensor, xs_neg_prob: Tensor, num_top_k: int):
    with torch.no_grad():
        unlable_index = (targets == 2).float()
        top_prob_index = torch.zeros_like(xs_neg_prob).bool()                                      
        rank_index = torch.argsort(xs_neg_prob, dim=-1)       
        top_prob_index[:, rank_index[:num_top_k]] = True
        
        return torch.logical_and(unlable_index, top_prob_index)

def get_neighbor_index(targets: Tensor, adj_mat: Tensor):
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
        anno_pos_index = (targets == 1).unsqueeze(1).float()
        neibors = torch.bmm(anno_pos_index, adj_mat).squeeze(1)
        return neibors == 1

def get_intersection_and_difference(A, B):
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

@loss("AsymmetricLoss")
class AsymmetricLoss(Loss):
    
    def __init__(self, cfg, clip=0.05) -> None:
        super().__init__("ASL")

        self.clip = clip
        self.gamma_pos = cfg.gamma_pos
        self.gamma_neg = cfg.gamma_neg
        self.gamma_unl = cfg.gamma_unl
        self.alpha_pos = cfg.alpha_pos
        self.alpha_neg = cfg.alpha_neg
        self.alpha_unl = cfg.alpha_unl
    
    def forward(self, logits, targets, targets_weights=None):
        
        device = logits.device

        # Activation
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)
        if targets_weights is None:
            targets_weights = torch.ones_like(targets, dtype=torch.float32).to(device)
        
        targets_pos = (targets == 1).float()
        targets_neg = (targets == 0).float()
        targets_unl = (targets == 2).float()

        # Loss calculation
        BCE_pos = self.alpha_pos * targets_pos * torch.log(torch.clamp(xs_pos, min=1e-8))
        BCE_neg = self.alpha_neg * targets_neg * torch.log(torch.clamp(xs_neg, min=1e-8))
        BCE_unl = self.alpha_unl * targets_unl * torch.log(torch.clamp(xs_neg, min=1e-8))

        BCE_loss = BCE_pos + BCE_neg + BCE_unl

        with torch.no_grad():
            asymmetric_w = torch.pow(1 - xs_pos * targets_pos - xs_neg * (targets_neg + targets_unl),
                                     self.gamma_pos * targets_pos + self.gamma_neg * targets_neg +
                                     self.gamma_unl * targets_unl)
        BCE_loss *= asymmetric_w

        # partial labels weights
        BCE_loss *= targets_weights

        return -BCE_loss.sum()
