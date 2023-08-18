import json
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from . import evaluation
from .utils import *
from .metric.map import MaskedmAP
from .metric.cmap import MaskedCmAP
from .metric.cv import ConstraintViolation


from .evaluation import Evaluation
from .attr_rec_eval import SingleClassMetric, GroupClassMetric, top_K_values, average_precision_score


@evaluation("hierarchical_attribute_recognition_evaluation")
class HierarchicalAttributeRecoginitionEvaluation(Evaluation):
    
    def __init__(
        self, 
        directory, 
        fpath_attr2idx,
        fpath_attr_headtail,
        f_hierarchy,
        threshold=0.5,
        exclude_atts=[],
        **kwargs
    ) -> None:
        
        self.hierarchy = np.load(f_hierarchy)
        
        with open(fpath_attr2idx, 'r') as f:
            self.attr2idx = json.load(f)
            self.idx2attr = {v:k for k, v in self.attr2idx.items()}

        # Read file that shows whether attribute is head/mid/tail.
        with open(fpath_attr_headtail, 'r') as f:
            self.attribute_head_tail = json.load(f)

        self.n_class = len(self.attr2idx)
        self.exclude_atts = exclude_atts
        self.threshold = threshold

        # Cache metric score for each class.
        self.score = {} # key: i_class -> value: all metrics.
        self.score_topk = {}

        super().__init__(directory)
    
    def build_metrics(self):
        self.metrics = [MaskedmAP(name="mAP")]
        # self.metrics = [ConstraintViolation(name='CV', adjacency_matrix=self.hierarchy), MaskedCmAP(name='cmAP', adjacency_matrix=self.hierarchy)]

    def reset_metrics(self):
        for m in self.metrics:
            m.reset()

    def __call__(self, dataloader, model):

        self.reset_metrics()
        
        model.eval()

        preds = []
        targets = []
        
        device = next(model.parameters()).device
        pbar = tqdm(dataloader)
        pbar.set_description(f'Evaluation')
        with torch.no_grad():
            for i, data in enumerate(pbar):
                target = data['t']
                data = load_to_device(data, device)
                pred = model(data).sigmoid()
                preds.append(pred.cpu())
                targets.append(target.cpu())
                for metric in self.metrics:
                    metric.update(preds=pred, gt_labels=target)
        
        for metric in self.metrics:
            metric.calculate_metric()
            self._result.update(metric.get_result())

        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

        scores_overall, scores_per_class = self.evaluate(preds, targets)
        scores_overall_topk, scores_per_class_topk = self.evaluate(preds, targets, threshold_type='topk')

        CATEGORIES = ['all', 'head', 'medium', 'tail']

        for category in CATEGORIES:
            
            category_result = {}
            category_result.update(mAP=f"{scores_per_class[category]['ap']:.4f}")
            
            perclass_result = {}
            for metric in ['recall', 'precision', 'f1', 'bacc']:
                if metric in scores_per_class[category]:
                    perclass_result[metric] = f"{scores_per_class[category][metric]:.4f}"
            category_result[f"Perclass-th{self.threshold}"] = perclass_result
            
            perclass_topk_result = {}
            for metric in ['recall', 'precision', 'f1']:
                if metric in scores_per_class_topk[category]:
                    perclass_topk_result[metric] = f"{scores_per_class_topk[category][metric]:.4f}"
            category_result[f"Perclass-top15"] = perclass_topk_result

            overall_result = {}
            for metric in ['recall', 'precision', 'f1', 'bacc']:
                if metric in scores_overall[category]:
                    overall_result[metric] = f"{scores_overall[category][metric]:.4f}"
            category_result[f"Overall-th{self.threshold}"] = overall_result
            
            overall_topk_result = {}
            for metric in ['recall', 'precision', 'f1']:
                if metric in scores_overall_topk[category]:
                    overall_topk_result[metric] = f"{scores_overall_topk[category][metric]:.4f}"

            category_result[f"Overall-top15"] = overall_topk_result
            self._result[category] = category_result

        self.print_result()
        self.save_result()
    
    def get_mAP(self):
        return float(self._result['all']['mAP'])

    def _clear_cache(self):
        self.score = {}
        self.score_topk = {}

    def get_attr_type(self, attr):
        """Finds type and subtype of a given attribute.
        """
        ty = 'other'
        subty = 'other'
        for x, L in self.attribute_type.items():
            if attr in L:
                subty = x
                break
        for x, L in self.attribute_parent_type.items():
            if subty in L:
                ty = x
                break
        return ty, subty

    def get_attr_head_tail(self, attr):
        """Finds whether attribute is in head/medium/tail group.
        """
        for group, L in self.attribute_head_tail.items():
            if attr in L:
                return group
        assert False, f"Can't find head/medium/tail group for {attr}"



    def evaluate(
        self,
        pred,
        gt_label,
        threshold_type='threshold'
    ):
        """Evaluates a prediction matrix against groundtruth label.
        Args:
        - pred:     prediction matrix [n_instance, n_class].
                    pred[i,j] is the j-th attribute score of instance i-th.
                    These scores should be from 0 -> 1.
        - gt_label: groundtruth label matrix [n_instances, n_class].
                    gt_label[i,j] = 1 if instance i is positively labeled with
                    attribute j, = 0 if it is negatively labeled, and = 2 if
                    it is unlabeled.
        - threshold_type: 'threshold' or 'topk'. 
                          Determines positive vs. negative prediction.
        """
        self._clear_cache()
        self.pred = pred
        self.gt_label = gt_label
        self.n_instance = self.gt_label.shape[0]

        # For topK metrics, we keep a version of the prediction matrix that sets
        # non-topK elements as 0 and topK elements as 1.

        P_topk = self.pred.copy()
        P_topk = np.apply_along_axis(top_K_values, 1, P_topk)
        P_topk[P_topk > 0] = 1
        self.pred_topk = P_topk

        all_groups = ['all', 'head', 'medium', 'tail']
        groups_overall = {
            k: GroupClassMetric(metric_type='overall')
            for k in all_groups
        }
        groups_per_class = {
            k: GroupClassMetric(metric_type='per-class')
            for k in all_groups
        }

        for i_class in range(self.n_class):
            attr = self.idx2attr[i_class]
            if attr in self.exclude_atts:
                continue

            class_metric = self.get_score_class(i_class, threshold_type=threshold_type)

            # Add to 'all' group.
            groups_overall['all'].add_class(class_metric)
            groups_per_class['all'].add_class(class_metric)

            imbalance_group = self.get_attr_head_tail(attr)
            groups_overall[imbalance_group].add_class(class_metric)
            groups_per_class[imbalance_group].add_class(class_metric)
    

        scores_overall = {}
        for group_name, group in groups_overall.items():
            scores_overall[group_name] = {
                'f1': group.get_f1(),
                'precision': group.get_precision(),
                'recall': group.get_recall(),
                'tnr': group.get_tnr(),
            }
        scores_per_class = {}
        for group_name, group in groups_per_class.items():
            scores_per_class[group_name] = {
                'ap': group.get_ap(),
                'f1': group.get_f1(),
                'precision': group.get_precision(),
                'recall': group.get_recall(),
                'bacc': group.get_bacc()
            }

        return scores_overall, scores_per_class
        
    def get_score_class(self, i_class, threshold_type='threshold'):
        """Computes all metrics for a given class.
        Args:
        - i_class: class index.
        - threshold_type: 'topk' or 'threshold'. This determines how a
        prediction is positive or negative.
        """
        if threshold_type == 'threshold':
            score = self.score
        else:
            score = self.score_topk
        if i_class in score:
            return score[i_class]

        if threshold_type == 'threshold':
            pred = self.pred[:,i_class].copy()
        else:
            pred = self.pred_topk[:,i_class].copy()
        gt_label = self.gt_label[:,i_class].copy()

        # Find instances that are explicitly labeled (either positive or negative).
        mask_labeled = (gt_label < 2)
        if mask_labeled.sum() == 0:
            # None of the instances have label for this class.
            # assert False, f"0 labeled instances for attribute {self.idx2attr[i_class]}"
            pass
        else:
            # Select only the labeled ones.
            pred = pred[mask_labeled]
            gt_label = gt_label[mask_labeled]

        if threshold_type == 'threshold':
            # Only computes AP when threshold_type is 'threshold'. This is because when
            # threshold_type is 'topk', pred is a binary matrix.
            ap = average_precision_score(gt_label, pred)

            # Make pred into binary matrix.
            pred[pred > self.threshold] = 1
            pred[pred <= self.threshold] = 0

        class_metric = SingleClassMetric(pred, gt_label)
        if threshold_type == 'threshold':
            class_metric.ap = ap

        # Cache results.
        score[i_class] = class_metric
        
        return class_metric
