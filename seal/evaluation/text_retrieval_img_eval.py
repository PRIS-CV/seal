import json
from typing import Any
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from . import evaluation
from .evaluation import Evaluation
from .utils import *
from .metric.recallatk import RecallAtK


@evaluation('text_retrieval_image_evaluation')
class TextRetrievalImageEvaluation(Evaluation):
    
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