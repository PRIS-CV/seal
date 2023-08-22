import pandas as pd
import torch
from torch import Tensor

from seal.models.loss.loss import Loss
from seal.models.loss import loss
from IPython import embed

@loss("PartialSelectiveLoss")
class PartialSelectiveLoss(Loss):

    def __init__(
        self, 
        gamma_pos, gamma_neg, gamma_unl,
        alpha_pos, alpha_neg, alpha_unl,
        partial_loss_mode = 'selective',
        prior_path = None,
        likelihood_topk = 5,
        prior_threshold = 0.5,
        targets_weights = None,
        clip=0.05,
        **kwargs
    ):
        super().__init__("psl")
        self.clip = clip
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.gamma_unl = gamma_unl
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.alpha_unl = alpha_unl
        self.prior_path =prior_path
        self.partial_loss_mode = partial_loss_mode
        self.likelihood_topk = likelihood_topk
        self.prior_threshold = prior_threshold
        self.targets_weights = targets_weights

        if self.partial_loss_mode == 'selective' and self.prior_path is not None:
            print("Prior file was found in given path.")
            df = pd.read_csv(self.prior_path)
            self.prior_classes = dict(zip(df.values[:, 0], df.values[:, 1]))
            print("Prior file was loaded successfully. ")

    def forward(self, logits, targets):

        # Positive, Negative and Un-annotated indexes
        targets_pos = (targets == 1).float()
        targets_neg = (targets == 0).float()
        targets_unl = (targets == 2).float()
        
        device = logits.device

        # Activation
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        prior_classes = None
        if hasattr(self, "prior_classes"):
            prior_classes = torch.tensor(list(self.prior_classes.values())).to(device)

        targets_weights = self.targets_weights
        targets_weights, xs_neg = edit_targets_parital_labels(targets=targets, targets_weights=targets_weights, xs_neg=xs_neg,
                                                              prior_classes=prior_classes, partial_loss_mode=self.partial_loss_mode,
                                                              likelihood_topk=self.likelihood_topk, prior_threshold=self.prior_threshold)

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
        

def negative_backprop_fun_jit(targets: Tensor, xs_neg_prob: Tensor, targets_weights: Tensor, num_top_k: int):
    with torch.no_grad():
        targets_flatten = targets.flatten()
        cond_flatten = torch.where(targets_flatten == 2)[0]
        targets_weights_flatten = targets_weights.flatten()
        xs_neg_prob_flatten = xs_neg_prob.flatten()
        ind_class_sort = torch.argsort(xs_neg_prob_flatten[cond_flatten])
        targets_weights_flatten[
            cond_flatten[ind_class_sort[:num_top_k]]] = 0


def edit_targets_parital_labels(targets, targets_weights, xs_neg, partial_loss_mode=None, prior_classes=None, likelihood_topk=5, prior_threshold=0.5):
    
    device = xs_neg.device
    
    # targets_weights is and internal state of AsymmetricLoss class. we don't want to re-allocate it every batch
    if partial_loss_mode is None:
        targets_weights = 1.0

    elif partial_loss_mode == 'negative':
        # set all unsure targets as negative
        targets_weights = 1.0

    elif partial_loss_mode == 'ignore':
        # remove all unsure targets (targets_weights=0)
        targets_weights = torch.ones(targets.shape, device=device)
        targets_weights[targets == 2] = 0

    elif partial_loss_mode == 'ignore_normalize_classes':
        # remove all unsure targets and normalize by Durand et al. https://arxiv.org/pdf/1902.09720.pdfs
        alpha_norm, beta_norm = 1, 1
        targets_weights = torch.ones(targets.shape, device=device)
        n_annotated = 1 + torch.sum(targets != 2, axis=1)    # Add 1 to avoid dividing by zero

        g_norm = alpha_norm * (1 / n_annotated) + beta_norm
        n_classes = targets_weights.shape[1]
        targets_weights *= g_norm.repeat([n_classes, 1]).T
        targets_weights[targets == 2] = 0

    elif partial_loss_mode == 'selective':
        if targets_weights is None or targets_weights.shape != targets.shape:
            targets_weights = torch.ones(targets.shape, device=device)
        else:
            targets_weights[:] = 1.0
        num_top_k = likelihood_topk * targets_weights.shape[0]

        xs_neg_prob = xs_neg
        if prior_classes is not None:
            if prior_threshold:
                idx_ignore = torch.where(prior_classes > prior_threshold)[0]
                targets_weights[:, idx_ignore] = 0
                targets_weights += (targets != 2).float()
                targets_weights = targets_weights.bool()

        negative_backprop_fun_jit(targets, xs_neg_prob, targets_weights, num_top_k)

    return targets_weights, xs_neg
