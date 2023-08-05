import torch.nn as nn


class ALModel(nn.Module):
    r"""Attribute Learning Model.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self):
        raise NotImplementedError("Each attribute learning model should implement the forward method")

    def get_params(self):
        raise NotImplementedError("Each attribute learning model should implement the get_params method")
    