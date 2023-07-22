import torch.nn as nn


from info.models.layer.layer import Layer
from info.models.layer import layer
from info.models.layer.pwff import PositionWiseFeedForward
from info.models.attention.multi_head_att import MultiHeadAttention


@layer("CMA")
class CrossModalMHA(Layer):

    def __init__(self, cfg):
        super().__init__()

        self.d_model = cfg.d_model
        self.d_k = cfg.d_k
        self.d_v = cfg.d_v, 
        self.h = cfg.h
        self.d_ff = cfg.d_ff
        self.dropout = cfg.dropout
        self.self_att_module = cfg.self_att_module
        self.enc_att_module = cfg.enc_att_module
        self.self_att_module_kwargs = cfg.self_att_module_kwargs
        self.enc_att_module_kwargs = cfg.enc_att_module_kwargs

        self.enc_att = MultiHeadAttention(
            self.d_model, self.d_k, self.d_v, self.h, self.d_ff, dropout=self.dropout, 
            can_be_stateful=False,
            attention_module=self.enc_att_module,
            attention_module_kwargs=self.enc_att_module_kwargs
        )

        self.dropout2 = nn.Dropout(self.dropout)
        self.lnorm2 = nn.LayerNorm(self.d_model)
        self.pwff = PositionWiseFeedForward(self.d_model, self.d_ff, self.dropout)

    def forward(self, input, enc_output, mask_enc_att):
        
        # MHA + AddNorm
        enc_att = self.enc_att(input, enc_output, enc_output, mask_enc_att)
        enc_att = self.lnorm2(input + self.dropout2(enc_att))
        
        # FFN + AddNorm
        ff = self.pwff(enc_att)
        return ff
