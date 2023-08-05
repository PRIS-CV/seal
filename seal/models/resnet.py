import torch
import torch.nn as nn
from yacs.config import CfgNode as CN

from seal.models import almodel_config, almodel
from seal.models.model import ALModel
from seal.models.config import ALModelConfig
from seal.models.loss import build_loss
from seal.models.backbone import build_backbone
from seal.models.utils import add_weight_decay_and_lr, ROIAlign


@almodel_config("RN50 Config")
class RN50Config(ALModelConfig):
    
    def __init__(self, name="RN50 Config"):
        super().__init__(name)
        self.cfg.loss = "PartialSelectiveLoss"
        self.cfg.prior_path = ""
        self.cfg.partial_loss_mode = "ignore"   #  modes: ['negative', 'ignore', 'ignore_normalize_classes', 'selective']
        self.cfg.likelihood_topk = 5
        self.cfg.prior_threshold = 0.5
        self.cfg.backbone = "resnet50"
        self.cfg.pretrained = True
        self.cfg.num_classes = 620
        self.cfg.num_objects = 2260
        self.cfg.fast_params = True
        self.cfg.fast_lr = 7e-4
        self.cfg.base_lr = 1e-5
        self.cfg.weight_decay = 1e-4
        self.cfg.f_obj_embs = ""
        self.cfg.d_obj = 2260
        self.cfg.gamma_pos = 0
        self.cfg.gamma_neg = 0
        self.cfg.gamma_unl = 0
        self.cfg.gamma_can = 0
        self.cfg.alpha_pos = 1
        self.cfg.alpha_neg = 1
        self.cfg.alpha_unl = 1
        self.cfg.alpha_can = 1


@almodel("RN50")
class RN50(ALModel):
    
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.backbone = build_backbone(cfg.backbone)(cfg.pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, cfg.num_classes)
        self.loss_fn = build_loss(cfg.loss)(cfg)
        self.num_classes = cfg.num_classes

    def fast_params(self):
        modules_name = ['classifier']
        modules = [getattr(self, m) for m in modules_name]
        fast_params = []
        for m_name, m in zip(modules_name, modules):
            fast_params.extend([m_name + "." + p_name for p_name, _ in m.named_parameters()])
        return fast_params

    def get_params(self, show_detail):
        if self.cfg.fast_params:
            params = add_weight_decay_and_lr(
                self, 
                self.cfg.weight_decay, 
                self.cfg.fast_lr, 
                self.cfg.base_lr, show_detail=show_detail)
        else:
            params = [{"params": [p for p in self.parameters() if p.requires_grad]}]
        return params
    
    def get_optimizer(self, show_detail=False):
        params = self.get_params(show_detail=show_detail)
        return torch.optim.Adam(params=params, lr=self.cfg.base_lr, weight_decay=0)
    
    def train_model(self, data):
        target = data['t']
        pred = self.infer(data).float()
        return self.compute_loss(pred, target)

    def compute_loss(self, pred, target):
        loss = self.loss_fn(pred, target)
        loss_dict = {self.loss_fn.name: loss.item()}
        return loss, loss_dict
    
    def infer(self, data):
        img = data['i']
        x = self.backbone(img)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        pred = self.classifier(x)
        return pred
    
    def get_visual_feature(self, data):
        img = data['i']
        x = self.backbone(img)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, data):
        if self.training:
            return self.train_model(data)
        else:
            return self.infer(data)


@almodel("RN50-OBJ")
class RN50_OBJ(RN50):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        
        self.obj_encoder = nn.Linear(cfg.d_obj, 2048)
        self.register_buffer("obj_embs", torch.load(cfg.f_obj_embs))

    def fast_params(self):
        modules_name = ['classifier', 'obj_encoder']
        modules = [getattr(self, m) for m in modules_name]
        fast_params = []
        for m_name, m in zip(modules_name, modules):
            fast_params.extend([m_name + "." + p_name for p_name, _ in m.named_parameters()])
        return fast_params

    def get_visual_feature(self, data):
        img = data['i']
        obj_index = data['o']
        obj_emb = self.obj_embs[obj_index]
        x = self.backbone(img)
        mask = self.obj_encoder(obj_emb)
        x = torch.sigmoid(mask).unsqueeze(2).unsqueeze(2) * x 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def infer(self, data):
        img = data['i']
        obj_index = data['o']
        obj_emb = self.obj_embs[obj_index]
        x = self.backbone(img)
        mask = self.obj_encoder(obj_emb)
        x = torch.sigmoid(mask).unsqueeze(2).unsqueeze(2) * x 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        pred = self.classifier(x)
        return pred


@almodel("RN101-OBJ")
class RN101_OBJ(RN50_OBJ):
    
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

