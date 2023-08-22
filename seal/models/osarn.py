"""Acknowledgement: This code is modified from Query2Label.
Paper: https://arxiv.org/abs/2107.10834 
Code: https://github.com/SlongLiu/query2labels
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import RGCNConv

from . import almodel
from .model import ALModel
from .loss import build_loss
from .utils import add_weight_decay_and_lr
from .backbone import build_backbone
from .decoder.osfm import build_osfm
from typing import List




class GroupWiseLinear(nn.Module):

    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x
    

class RelationGCN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations) -> None:
        super().__init__()

        self.gconv1 = RGCNConv(in_channels=input_dim, out_channels=hidden_dim, num_relations=num_relations)
        self.act = nn.LeakyReLU(0.2)
        self.norm1 = nn.LayerNorm(hidden_dim) 
        self.gconv2 = RGCNConv(in_channels=hidden_dim, out_channels=output_dim, num_relations=num_relations)
        self.norm2 = nn.LayerNorm(output_dim) 

    def forward(self, x, edge_index, edge_type):
        out = self.gconv1(x, edge_index=edge_index, edge_type=edge_type)
        out = self.act(out)
        out = self.norm1(out)
        out = self.gconv2(out, edge_index=edge_index, edge_type=edge_type)
        out = self.norm2(out)
        return out



@almodel("ObjectSpcAttRelNet")
class ObjectSpcAttRelNet(ALModel):
    
    def __init__(self, backbone, transformer, num_classes, rem, optimizer, loss, obj_spc, f_attr_embs, f_obj_embs, **kwargs) -> None:
        super().__init__()

        self.backbone = build_joiner(**backbone)
        self.transformer = build_osfm(obj_spc=obj_spc, **transformer)
        self.optim_set = optimizer
        hidden_dim = self.transformer.d_model
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        self.obj_encoder = nn.Linear(512, hidden_dim)
        self.fc = GroupWiseLinear(num_classes, hidden_dim, bias=True)

        self.loss_fn = build_loss(loss['name'])(**loss)

        self.remodule = RelationGCN(input_dim=512, hidden_dim=1024, output_dim=2048, num_relations=rem['n_type'])
        self.modules_name = ["transformer", "fc", "input_proj", "remodule"]
        self.register_buffer('query_embed', torch.load(f_attr_embs).float(), persistent=False)
        self.register_buffer('obj_embs', torch.load(f_obj_embs).float(), persistent=False)
        hmat = torch.from_numpy(np.load(rem["hir_adj"]) - np.identity(num_classes, np.int32)).float().t()


        edge_types = []
        all_relation_edge_index = []
        for i, rel in enumerate(['pos', 'neg', 'hir']):
            if rel == 'hir':
                hir_edge_index = self.get_edge_index_by_adj(hmat)
                hir_edge_type = torch.zeros(hir_edge_index.shape[1]).fill_(i)
                all_relation_edge_index.append(hir_edge_index)
                edge_types.append(hir_edge_type)
                print(f"Total Hir Relation NUM: {hmat.sum()}")

            elif rel == 'pos':
                
                pmat = torch.from_numpy(gen_A(t=rem["pos_threshold"], adj_file=rem["pos_adj"], num_file=rem["att_num"]))

                print("Exclue the edge in hierarchical relation matrix")
                print(f"Original Pos Relation NUM: {pmat.sum()}")
                pmat = ((pmat - hmat) > 0).float()
                print(f"Rest Pos Relation NUM: {pmat.sum()}")
        
                pos_edge_index = self.get_edge_index_by_adj(pmat)
                pos_edge_type = torch.zeros(pos_edge_index.shape[1]).fill_(i)
                all_relation_edge_index.append(pos_edge_index)
                edge_types.append(pos_edge_type)
            
            elif rel == 'neg':
                nmat = torch.from_numpy(gen_A(t=rem["neg_threshold"], adj_file=rem["neg_adj"], num_file=rem["att_num"]))
                neg_edge_index = self.get_edge_index_by_adj(nmat)
                neg_edge_type = torch.zeros(neg_edge_index.shape[1]).fill_(i)
                all_relation_edge_index.append(neg_edge_index)
                edge_types.append(neg_edge_type)
                print(f"Total Neg Relation NUM: {nmat.sum()}")

        edge_types = torch.cat(edge_types, dim=-1)
        all_relation_edge_index = torch.cat(all_relation_edge_index, dim=-1)

        self.register_buffer("edge_type", edge_types, persistent=False)
        self.register_buffer("edge_index", all_relation_edge_index, persistent=True)

    def get_edge_index_by_adj(self, adj):
        edge_index = adj.nonzero().t().contiguous().long()
        return edge_index
    
    def compute_loss(self, pred, target):
        loss = self.loss_fn(pred, target)
        loss_dict = {self.loss_fn.name: loss.item()}
        return loss, loss_dict

    def train_model(self, data):
        target = data['t']
        pred = self.infer(data).float()
        return self.compute_loss(pred, target)

    def forward(self, data):
        if self.training:
            return self.train_model(data)
        else:
            return self.infer(data)
        
    def fast_params(self):
        modules_name = self.modules_name
        modules = [getattr(self, m) for m in modules_name]
        fast_params = []
        for m_name, m in zip(modules_name, modules):
            fast_params.extend([m_name + "." + p_name for p_name, _ in m.named_parameters()])
        return fast_params

    def get_params(self, show_detail=True):
        params = add_weight_decay_and_lr(self, self.optim_set["weight_decay"], self.optim_set["fast_lr"], self.optim_set["base_lr"], show_detail=show_detail)
        return params

    def get_optimizer(self, show_detail=True):
        params = self.get_params(show_detail=show_detail)
        return torch.optim.AdamW(params=params, lr=self.optim_set["base_lr"], weight_decay=0)

    def infer(self, data):
        
        img = data['i']
        obj_index = data['o']
        
        src, pos = self.backbone(img)
        src, pos = src[-1], pos[-1]
        
        obj_token = None
        
        obj_emb = self.obj_embs[obj_index]
        obj_mask = self.obj_encoder(obj_emb)
        
        obj_token = obj_mask.unsqueeze(0)
        query_input = self.query_embed
            
        query_input = self.remodule(query_input, self.edge_index, self.edge_type)

        src = self.input_proj(src)  # [64, 2048, 7, 7]

        hs = self.transformer(src, query_input, pos, obj_emb=obj_token)[0] # B,K,d

        out = self.fc(hs[-1])
        return out



def gen_A(t, adj_file, num_file):
      
    _adj = np.load(adj_file)
    _nums = np.load(num_file)

    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[np.isnan(_adj)] = 0
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    return _adj


class Joiner(nn.Sequential):
    
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)


    def forward(self, input: Tensor):
        xs = self[0](input)
        out: List[Tensor] = []
        pos = []
        if isinstance(xs, dict):
            for name, x in xs.items():
                out.append(x)
                # position encoding
                pos.append(self[1](x).to(x.dtype))
        else:
            # for swin Transformer
            out.append(xs)
            pos.append(self[1](xs).to(xs.dtype))
        return out, pos



class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, maxH=30, maxW=30):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.maxH = maxH
        self.maxW = maxW
        pe = self._gen_pos_buffer()
        self.register_buffer('pe', pe)

    def _gen_pos_buffer(self):
        _eyes = torch.ones((1, self.maxH, self.maxW))
        y_embed = _eyes.cumsum(1, dtype=torch.float32)
        x_embed = _eyes.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, input: Tensor):
        x = input
        return self.pe.repeat((x.size(0),1,1,1))


def build_position_encoding(hidden_dim, position_embedding, img_size):

    N_steps = hidden_dim // 2
    downsample_ratio = 32

    if position_embedding in ('v2', 'sine'):

        assert img_size % 32 == 0, "args.img_size ({}) % 32 != 0".format(img_size)
        position_embedding = PositionEmbeddingSine(
            N_steps, normalize=True, maxH=img_size // downsample_ratio, maxW=img_size // downsample_ratio)
        
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding


def build_joiner(name, hidden_dim, position_embedding, img_size, pretrained, pretrained_weight):

    pos_embedding = build_position_encoding(hidden_dim, position_embedding, img_size)
    
    backbone = build_backbone(name)(pretrained=pretrained, pretrained_weight=pretrained_weight)
    backbone.forward = backbone.forward_features
    bb_num_channels = backbone.embed_dim * 8
    del backbone.avgpool
    del backbone.head

    model = Joiner(backbone, pos_embedding)
    model.num_channels = bb_num_channels
    return model
