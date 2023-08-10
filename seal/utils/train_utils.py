import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

import os
import os.path as op
import pandas as pd

from seal.utils import train_util
from seal.utils.utils import load_to_device
from seal.models.loss.psl import PartialSelectiveLoss
from seal.config import ConfigPlaceHolder


@train_util("train_one_epoch")
def train_one_epoch(model, dataloader, optimizer, epoch, epoch_num, device, amp=True, use_wandb=False):
    if amp:
        scaler = GradScaler()
    pbar = tqdm(dataloader)
    pbar.set_description(f'Epoch: {epoch + 1} / {epoch_num} Training')
    model.train()
    for i, data in enumerate(pbar):
        optimizer.zero_grad()
        data = load_to_device(data, device)
        if amp:
            with autocast():
                loss, loss_dict = model(data)
            scaler.scale(loss).backward()  
            scaler.step(optimizer)  
            scaler.update()  
        else:
            loss, loss_dict = model(data)
            # model.zero_grad()
            loss.backward()
            optimizer.step()
        lr_dict = {f"param-group-{i}": pg['lr'] for i, pg in enumerate(optimizer.state_dict()['param_groups'])}
        loss_dict.update(lr_dict)
        pbar.set_postfix(loss_dict)
        if use_wandb:
            wandb.log(loss_dict)
        # if i == 10:
        #     print('break')
        #     break

@train_util("compute_prior")
def compute_proir(model_cfg, model, dataloader, optimizer, epoch, epoch_num, device, prior, amp=True):
    loss_fn = PartialSelectiveLoss(model_cfg)
    if amp:
        scaler = GradScaler()
    pbar = tqdm(dataloader)
    pbar.set_description(f'Epoch: {epoch + 1} / {epoch_num} Training')
    model.train()
    for i, data in enumerate(pbar):
        optimizer.zero_grad()
        data = load_to_device(data, device)
        if amp:
            with autocast():
                output = model.infer(data).float()
                loss = loss_fn(output, data['t'])
                loss_dict = {"Ignore Mode Loss": loss.item()}
            scaler.scale(loss).backward()  
            scaler.step(optimizer)  
            scaler.update()  
        else:
            output = model.infer(data)
            loss = loss_fn(output, data['t'])
            loss_dict = {"Ignore Mode Loss": loss.item()}
            loss.backward()
            optimizer.step()
        if epoch > 1:
            prior.update(output)
        lr_dict = {f"param-group-{i}": pg['lr'] for i, pg in enumerate(optimizer.state_dict()['param_groups'])}
        loss_dict.update(lr_dict)
        pbar.set_postfix(loss_dict)


class ComputePrior:
    def __init__(self, classes, device, d_class_prior):
        self.classes = classes
        self.device = device
        n_classes = len(self.classes)
        self.sum_pred_train = torch.zeros(n_classes).to(device)
        self.sum_pred_val = torch.zeros(n_classes).to(device)
        self.cnt_samples_train,  self.cnt_samples_val = .0, .0
        self.avg_pred_train, self.avg_pred_val = None, None
        self.path_dest = d_class_prior

    def update(self, logits, training=True):
        with torch.no_grad():
            preds = torch.sigmoid(logits).detach()
            if training:
                self.sum_pred_train += torch.sum(preds, dim=0)
                self.cnt_samples_train += preds.shape[0]
                self.avg_pred_train = self.sum_pred_train / self.cnt_samples_train

            else:
                self.sum_pred_val += torch.sum(preds, dim=0)
                self.cnt_samples_val += preds.shape[0]
                self.avg_pred_val = self.sum_pred_val / self.cnt_samples_val

    def save_prior(self, name=""):

        print('Prior (train), first 5 classes: {}'.format(self.avg_pred_train[:5]))

        # Save data frames as csv files
        if not op.exists(self.path_dest):
            os.makedirs(self.path_dest)

        df_train = pd.DataFrame({"Classes": list(self.classes.keys()),
                                 "avg_pred": self.avg_pred_train.cpu()})
        df_train.to_csv(path_or_buf=os.path.join(self.path_dest, f"{name}_train_avg_preds.csv"),
                        sep=',', header=True, index=False, encoding='utf-8')

        if self.avg_pred_val is not None:
            df_val = pd.DataFrame({"Classes": list(self.classes.keys()),
                                   "avg_pred": self.avg_pred_val.cpu()})
            df_val.to_csv(path_or_buf=os.path.join(self.path_dest, f"{name}_val_avg_preds.csv"),
                          sep=',', header=True, index=False, encoding='utf-8')

    def get_top_freq_classes(self, n_top=10):
        top_idx = torch.argsort(-self.avg_pred_train.cpu())[:n_top]
        top_classes = np.array(list(self.classes.keys()))[top_idx]
        print('Prior (train), first {} classes: {}'.format(n_top, top_classes))
