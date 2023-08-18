import torch
import numpy as np
from sklearn.metrics import average_precision_score

from . import metric
from .metric import Metric


@metric("mAP")
class mAP(Metric):

    def __init__(self, name="mAP") -> None:
        super().__init__(name)
        self.reset()
        
    def update(self, preds, gt_labels, **kwargs):
        if isinstance(preds, np.ndarray):
            pass
        elif isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        else:
            raise ValueError("preds must be either numpy.ndarray or torch.Tensor")

        self.preds.append(preds)
        self.gt_labels.append(gt_labels)

    def calculate_metric(self):
        
        aps = []

        preds = np.vstack(self.preds)
        gt_labels = np.vstack(self.gt_labels)

        for i in range(gt_labels.shape[1]):
            
            target_class = gt_labels[:, i]
            pred_class = preds[:, i]
            
            ap = average_precision_score(target_class, pred_class)
            aps.append(ap)

        mAP = np.array(aps).mean()

        self._result[self.name] = mAP
    
    def reset(self):
        self.preds = []
        self.gt_labels = []
        self._result = {}


@metric("MaskedmAP")
class MaskedmAP(Metric):

    def __init__(self, name="MaskedmAP") -> None:
        super().__init__(name)
        self.reset()

    def calculate_metric(self):
        aps = []
        
        preds = np.vstack(self.preds)
        gt_labels = np.vstack(self.gt_labels)
        
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

        mAP = np.array(aps).mean()

        self._result[self.name] = mAP

    def update(self, preds, gt_labels, **kwargs):
        if isinstance(preds, np.ndarray):
            pass
        elif isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        else:
            raise ValueError("preds must be either numpy.ndarray or torch.Tensor")

        self.preds.extend(preds)
        self.gt_labels.extend(gt_labels)
    
    def reset(self):
        self.preds = []
        self.gt_labels = []
        self._result = {}
