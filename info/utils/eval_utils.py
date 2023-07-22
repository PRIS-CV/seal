import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from info.utils import eval_util
from info.utils.utils import load_to_device


@eval_util("vaw_eval_util")
def vaw_eval(model, dataloader, evaluator, device):
    model.eval()
    preds = []
    targets = []
    pbar = tqdm(dataloader)
    pbar.set_description(f'Testing')
    with torch.no_grad():
        for i, data in enumerate(pbar):
            # import copy
            # data_cp = copy.deepcopy(data)
            # del data
            # data = data_cp
            target = data['t']
            data = load_to_device(data, device)
            pred = model(data)
            preds.append(torch.sigmoid(pred).cpu().detach())
            targets.append(target)

    preds = torch.cat(preds).numpy()
    gt_label = torch.cat(targets).numpy()
    # import ipdb; ipdb.set_trace()

    scores_overall, scores_per_class = evaluator.evaluate(preds, gt_label)
    scores_overall_topk, scores_per_class_topk = evaluator.evaluate(
        preds, gt_label, threshold_type='topk')
    
    CATEGORIES = ['all', 'head', 'medium', 'tail'] + \
        list(evaluator.attribute_parent_type.keys())
    # CATEGORIES = ['all']

    for category in CATEGORIES:
        print(f"----------{category.upper()}----------")
        print(f"mAP: {scores_per_class[category]['ap']:.4f}")
        
        print("Per-class (threshold 0.5):")
        for metric in ['recall', 'precision', 'f1', 'bacc']:
            if metric in scores_per_class[category]:
                print(f"- {metric}: {scores_per_class[category][metric]:.4f}")
        print("Per-class (top 15):")
        for metric in ['recall', 'precision', 'f1']:
            if metric in scores_per_class_topk[category]:
                print(f"- {metric}: {scores_per_class_topk[category][metric]:.4f}")
    
        print("Overall (threshold 0.5):")
        for metric in ['recall', 'precision', 'f1', 'bacc']:
            if metric in scores_overall[category]:
                print(f"- {metric}: {scores_overall[category][metric]:.4f}")
        print("Overall (top 15):")
        for metric in ['recall', 'precision', 'f1']:
            if metric in scores_overall_topk[category]:
                print(f"- {metric}: {scores_overall_topk[category][metric]:.4f}")

    return scores_per_class['all']['ap']


@eval_util("cocoa_eval_util")
def cocoa_eval(model, dataloader, evaluator, device, mask_label=False):
    model.eval()
    preds = []
    targets = []
    pbar = tqdm(dataloader)
    pbar.set_description(f'Testing')
    with torch.no_grad():
        for i, data in enumerate(pbar):
            target = data['h_0']
            data = load_to_device(data, device)
            pred = model(data)
            preds.append(torch.sigmoid(pred).cpu().detach())
            targets.append(target)

    preds = torch.cat(preds).numpy()
    gt_labels = torch.cat(targets).numpy()

    if not mask_label:
        gt_labels[gt_labels == 2] = 0

    aps = []
    res = ""

    for i in range(gt_labels.shape[1]):
        
        target_class = gt_labels[:, i]
        pred_class = preds[:, i]
        mask_labeled = (target_class != 2)
        if mask_labeled.sum() == 0:
            # None of the instances have label for this class.
            # assert False, f"0 labeled instances for attribute {self.idx2attr[i_class]}"
            pass
        else:
            # Select ony the labeled ones.
            pred_class = pred_class[mask_labeled]
            target_class = target_class[mask_labeled]
            ap = average_precision_score(target_class, pred_class)
        aps.append(ap)
        if i < 10:
            res += f"{i}-AP: {ap:.4f} \n"
    mAP = np.array(aps).mean()
    res += f"Overall mAP: {mAP:.4f} \n"
    print(res)
    return mAP


@eval_util("ml_eval_util")
def multilabel_eval(model, dataloader, evaluator, device, mask_label=True):
    model.eval()
    preds = []
    targets = []
    pbar = tqdm(dataloader)
    pbar.set_description(f'Testing')
    with torch.no_grad():
        for i, data in enumerate(pbar):
            target = data['h_0']
            data = load_to_device(data, device)
            pred = model(data)
            preds.append(torch.sigmoid(pred).cpu().detach())
            targets.append(target)

    preds = torch.cat(preds).numpy()
    gt_labels = torch.cat(targets).numpy()

    if not mask_label:
        gt_labels[gt_labels == 2] = 0

    aps = []
    res = ""
    for i in range(gt_labels.shape[1]):
        
        target_class = gt_labels[:, i]
        pred_class = preds[:, i]
        mask_labeled = (target_class != 2)
        if mask_labeled.sum() == 0:
            # None of the instances have label for this class.
            # assert False, f"0 labeled instances for attribute {self.idx2attr[i_class]}"
            pass
        else:
            # Select ony the labeled ones.
            pred_class = pred_class[mask_labeled]
            target_class = target_class[mask_labeled]
            ap = average_precision_score(target_class, pred_class)
        aps.append(ap)
        if i < 10:
            res += f"{i}-AP: {ap:.4f} \n"
    mAP = np.array(aps).mean()
    res += f"Overall mAP: {mAP:.4f} \n"
    print(res)
    return mAP
