"""Acknowledgement: This contrastive loss is modified from mmf.
Code: https://github.com/facebookresearch/mmf.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Dict

from . import loss
from .loss import Loss
from ...utils.distributed import gather_tensor_along_batch_with_backward, get_rank

@loss("ContrastiveLoss")
class ContrastiveLoss(Loss):
    """
    This is a generic contrastive loss typically used for pretraining. No modality
    assumptions are made here.
    """

    def __init__(self, temperature: Union[float, Tensor] = 1.0):
        super().__init__(name="ContrastiveLoss")
        self.temperature = temperature

    def forward(self, sample_list: Dict[str, Tensor], model_output: Dict[str, Tensor]):
        embedding_1 = model_output["scores"]
        embedding_2 = model_output["targets"]

        # FIXME: maybe we can do not use val loss
        if embedding_1.size(0) != embedding_2.size(0):
            return torch.tensor(0.0, device=embedding_1.device)

        per_gpu_batch_size = embedding_1.size(0)

        embedding_1_all_gpus = gather_tensor_along_batch_with_backward(embedding_1)
        embedding_2_all_gpus = gather_tensor_along_batch_with_backward(embedding_2)

        logits_1 = (
            torch.matmul(embedding_1, embedding_2_all_gpus.transpose(0, 1))
            / self.temperature
        )
        logits_2 = (
            torch.matmul(embedding_2, embedding_1_all_gpus.transpose(0, 1))
            / self.temperature
        )
        labels = per_gpu_batch_size * get_rank() + torch.arange(
            per_gpu_batch_size, device=embedding_1.device
        )

        loss_1 = F.cross_entropy(logits_1, labels)
        loss_2 = F.cross_entropy(logits_2, labels)

        return (loss_1 + loss_2) / 2
    