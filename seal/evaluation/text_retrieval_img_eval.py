import json
from typing import Any
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from . import evaluation
from .evaluation import Evaluation
from .utils import *
from .metric.recallatk import RecallAtK
from .metric.group_diversity import GroupDiversity


@evaluation('tRi_accuracy_evaluation')
class TRIAccuracyEvaluation(Evaluation):
    
    def __init__(self, directory, **kwargs):
        super().__init__(directory)

    def build_metrics(self):
        self.metrics = [RecallAtK()]
    
    def __call__(self, dataloader, model, classnames: list, K: int, *args, **kwargs):
        scores = []
        model.eval()
        text_features = model.encode_text(classnames)
        device = next(model.parameters()).device
        pbar = tqdm(dataloader)
        pbar.set_description(f'Evaluation')
        with torch.no_grad():
            for i, data in enumerate(pbar):
                data = load_to_device(data, device)
                image_features = model.encode_image(data['i'])
                batch_scores = text_features @ image_features.T
                scores.append(batch_scores)

        scores = torch.cat(scores, dim=1)
        values, indexes = scores.topk(K, dim=1)
        indexes = indexes.cpu().numpy()
        values = values.cpu().numpy()
        
        return values, indexes

    def evaluate(self):

        for metric in self.metrics:
            metric.calculate_metric()
            self._result.update(metric.get_result())
            metric.reset()
        
        self.print_result()
        self.save_result()

    def get_mAP(self):
        return self._result["recall@k"]


@evaluation('tRi_group_diversity_evaluation')
class TRIGroupDiversityEvaluation(Evaluation):
    r"""TRIGroupDiversityEvaluation is evaluation class used to evaluate the group diversity of image retrieval by text embedding of certain context. It use entropy to measure its diversity.
    """
    
    def __init__(self, directory, **kwargs):
        super().__init__(directory)

    def build_metrics(self):
        self.group_diversity = GroupDiversity(name='GroupDiversity')
    
    def __call__(self, dataloader, model, groups: list, K: int, *args, **kwargs):
        
        model.eval()
        text_features = model.encode_text(groups)
        device = text_features.device
        
        groups_indexes = {}
        groups_values = {}
        groups_ids = {}

        for i, g in enumerate(groups):
            scores = []
            ids = []
            dataloader.dataset.change_group(g)
            pbar = tqdm(dataloader)
            pbar.set_description(f'Retrieval group {g}')
            with torch.no_grad():
                for _, data in enumerate(pbar):
                    data = load_to_device(data, device)
                    image_features = model.encode_image(data['i'])
                    batch_scores = text_features[i] @ image_features.T
                    batch_ids = data["group_id"]
                    scores.append(batch_scores)
                    ids.append(batch_ids)

            scores = torch.cat(scores, dim=0)
            ids = torch.cat(ids, dim=0)

            group_size = scores.shape[0]

            if group_size >= K:
                values, topk_indexes = scores.topk(K, dim=0)
            else:
                values, topk_indexes = scores.topk(group_size, dim=0)

            groups_ids[g] = ids[topk_indexes].cpu().numpy()

            groups_indexes[g] = topk_indexes.cpu().numpy()
            groups_values[g] = values.cpu().numpy()

        self.evaluate(groups_ids)

        return groups_values, groups_indexes

    def evaluate(self, groups_ids):

        self.group_diversity.calculate_metric(groups_ids)
        self._result.update(self.group_diversity.get_result())
        self.group_diversity.reset()
        
        self.print_result()
        self.save_result()

    def get_mAP(self):
        return self._result["recall@k"]