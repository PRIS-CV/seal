import os.path as op
import torch
import torch.nn as nn
from torch.nn import Module, ModuleList
from info.models.layer.pwff import PositionWiseFeedForward
from info.models.attention.multi_head_att import MultiHeadAttention


class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        
        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_enc_att):
        # MHA + AddNorm
        enc_att = self.enc_att(input, enc_output, enc_output, mask_enc_att)
        enc_att = self.lnorm2(input + self.dropout2(enc_att))
        # FFN + AddNorm
        ff = self.pwff(enc_att)
        return ff


class PAEncoder(nn.Module):
    def __init__(self, num_prototypes, f_prototypes, d_in=1024, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        """
        Args:
            num_centers: 400, 800, 2000
        """
        super(PAEncoder, self).__init__()

        # assert all(x > y for x, y in zip(num_prototypes, num_prototypes[1:])), "The sequence should be decreasing monotonically."
        assert op.exists(f_prototypes), "You should generated Atrributes Prototype Tree First."
        
        self.num_prototypes = num_prototypes
        self.f_prototypes = f_prototypes

        self.embed = nn.Linear(d_in, d_model)
        self.dropout = nn.Dropout(p=.1)
        self.fuse_ln = nn.LayerNorm(d_model)
        self.layers = ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                          enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                          enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(len(num_prototypes))])

        all_prototypes = torch.load(self.f_prototypes)
        self.concept_protos = [all_prototypes[f'attr-prototype-{num}'] for num in self.num_prototypes]

    def forward(self, grid):
        """
        Input:
            grid_features: [bsz, 49, 1024]
            protos:        list of prototypes [[800, 1024], [800, 1024], [2000, 1024]]
        Return:
            out: [bzs, 49, 512]
        """

        out = self.embed(grid)
        for i, l in enumerate(self.layers):
            concepts = self.concept_protos[i].to(grid.device).unsqueeze(0).repeat(out.shape[0], 1, 1)  # [bsz, num, dim]
            mask_concepts = (torch.sum(concepts, -1) == 0).unsqueeze(1).unsqueeze(1)    # [num, 1, 1, dim]
            out = l(out, concepts, mask_concepts)

        return out
