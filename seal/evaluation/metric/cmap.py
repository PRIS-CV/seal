import numpy as np
import torch
from sklearn.metrics import average_precision_score

from . import metric
from .metric import Metric
from .utils import get_constr_out


@metric("MaskedCmAP")
class MaskedCmAP(Metric):

    def __init__(self, name, adjacency_matrix) -> None:
        super().__init__(name)

        if isinstance(adjacency_matrix, np.ndarray):
            self.adjacency_matrix = torch.from_numpy(adjacency_matrix)
        elif isinstance(adjacency_matrix, torch.Tensor):
            self.adjacency_matrix = adjacency_matrix
        else:
            raise ValueError("adjacency_matrix must be numpy.ndarray")
        
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
            _preds = torch.from_numpy(preds)
            _preds = get_constr_out(_preds, self.adjacency_matrix).numpy()
        elif isinstance(preds, torch.Tensor):
            _preds = get_constr_out(preds.cpu().clone(), self.adjacency_matrix).numpy()
        else:
            raise ValueError("preds must be either numpy.ndarray or torch.Tensor")

        self.preds.extend(_preds)
        self.gt_labels.extend(gt_labels)

    def reset(self):
        self.preds = []
        self.gt_labels = []
        self._result = {}



@metric("CmAP")
class CmAP(Metric):

    def __init__(self, name, adjacency_matrix) -> None:
         super().__init__(name)

         self.adjacency_matrix = adjacency_matrix

    def update(self, preds, gt_labels, **kwargs):
        
        if isinstance(preds, np.ndarray):
            _preds = torch.from_numpy(preds)
            _preds = get_constr_out(_preds, self.adjacency_matrix).numpy()
        elif isinstance(preds, torch.Tensor):
            _preds = get_constr_out(preds.cpu().clone(), self.adjacency_matrix).numpy()
        else:
            raise ValueError("preds must be either numpy.ndarray or torch.Tensor")

        self.preds.extend(_preds)
        self.gt_labels.extend(gt_labels)

    def calculate_metric(self, preds, gt_labels, **kwargs):
        
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
