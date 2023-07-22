import torch.nn as nn


class Loss(nn.Module):

    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    def forward(self):
        raise NotImplementedError("Each loss function should implement forward method.")
    