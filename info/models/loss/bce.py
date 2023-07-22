import torch

from info.models.loss.loss import Loss
from info.models.loss import loss


@loss("WeightedBCELoss")
class WeightedBCELoss(Loss):
    
    def __init__(self, pos_weight: float = 1.0, neg_weight: float = 1.0, unl_weight: float = 0.1):
        super().__init__("wbce")
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.unl_weight = unl_weight
        self.reduction = 2

    def get_weight(self, target):
        weight = torch.ones_like(target)
        weight[target == 1] == self.pos_weight
        weight[target == 0] == self.neg_weight
        weight[target == 2] == self.unl_weight
        return weight

    def forward(self, pred, target):
        weight = self.get_weight(target)
        target[target == 2] = 0
        loss = torch.binary_cross_entropy_with_logits(input=pred, target=target, weight=weight, reduction=self.reduction) / target.shape[0]
        return loss
