import torch
import torch.nn as nn
from yacs.config import CfgNode as CN

from seal.models import almodel_config, almodel
from seal.models.model import ALModel
from seal.models.config import ALModelConfig
from seal.models.loss import build_loss
from seal.models.backbone import build_backbone
from seal.models.utils import add_weight_decay_and_lr, ROIAlign


@almodel("ResNetOBJ")
class ResNetOBJ(ALModel):

    def __init__(self, d_obj: int, f_obj_embs: str, num_classes:int, loss: dict, backbone: dict, optimizer: dict) -> None:
        
        super().__init__()
        
        self.backbone = build_backbone(backbone['name'])(**backbone)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, num_classes)
        self.loss_fn = build_loss(loss['name'])(**loss)
        self.num_classes = num_classes
        self.obj_encoder = nn.Linear(d_obj, 2048)
        self.register_buffer("obj_embs", torch.load(f_obj_embs))
        self.optimizer_settings = optimizer

    def fast_params(self):
        modules_name = self.optimizer_settings['fast_params']
        modules = [getattr(self, m) for m in modules_name]
        fast_params = []
        for m_name, m in zip(modules_name, modules):
            fast_params.extend([m_name + "." + p_name for p_name, _ in m.named_parameters()])
        return fast_params

    def get_params(self, show_detail):
        if len(self.optimizer_settings['fast_params']) > 0:
            params = add_weight_decay_and_lr(
                self, 
                self.optimizer_settings['weight_decay'],
                self.optimizer_settings['fast_lr'],
                self.optimizer_settings['base_lr'], 
                show_detail=show_detail)
        else:
            params = [{"params": [p for p in self.parameters() if p.requires_grad]}]
        return params

    def get_optimizer(self, show_detail=False):
        params = self.get_params(show_detail=show_detail)
        return torch.optim.Adam(params=params, lr=self.optimizer_settings['base_lr'], weight_decay=0)
    
    def train_model(self, data):
        target = data['t']
        pred = self.infer(data).float()
        return self.compute_loss(pred, target)

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
    
    def compute_loss(self, pred, target):
        loss = self.loss_fn(pred, target)
        loss_dict = {self.loss_fn.name: loss.item()}
        return loss, loss_dict
    
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

