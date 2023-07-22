import torch.nn as nn


class Encoder(nn.Module):
    
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self):
        raise NotImplementedError("")