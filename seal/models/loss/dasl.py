"""Acknowledgement: This code is modified from DualCoop
Paper: 
Code:
"""


import torch
import torch.nn as nn
import logging

from . import loss
from .loss import Loss


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@loss("DualAsymmetricLoss")
class DualAsymmetricLoss(Loss):
    def __init__(
        self, 
        name="DualAsymmetricLoss", 
        gamma_neg=2, gamma_pos=0, gamma_unl=4,
        alpha_pos=1, alpha_neg=1, alpha_unl=1,
        clip=0.05, 
        eps=1e-6
    ):
        
        super().__init__(name)

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.gamma_unl = gamma_unl
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.alpha_unl = alpha_unl
        self.clip = clip
        self.eps = eps
        self.softmax = nn.Softmax(dim=1)

        logger.info(f"Using DualAsymmetricLoss with config: {self.config}")

    @property
    def config(self):
        config = f"\ngamma_neg: {self.gamma_neg}" \
                f"\ngamma_pos: {self.gamma_pos}" \
                f"\ngamma_unl: {self.gamma_unl}" \

        return config

    def forward(self, x, targets):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_softmax = self.softmax(x)
        xs_pos = x_softmax[:, 1, :]
        xs_neg = x_softmax[:, 0, :]
        
        xs_pos = xs_pos.flatten(1)
        xs_neg = xs_neg.flatten(1)

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

        return -BCE_loss.sum()