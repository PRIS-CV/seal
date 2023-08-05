import torch

from optimizer import optimizer


@optimizer("Adam")
def Adam(cfg, params):
    return torch.optim.Adam(params=params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)

@optimizer("AdamW")
def AdamW(cfg, params):
    return torch.optim.AdamW(params=params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)

@optimizer("SGD")
def SGD(cfg, params):
    return torch.optim.SGD(params=params, lr=cfg.base_lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
