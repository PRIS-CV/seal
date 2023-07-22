import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from info.utils.metrics import GroupClassMetric, SingleClassMetric, average_precision
from info.utils.utils import top_K_values


class VAWEvaluator(object):
    def __init__(
        self,
        fpath_attr2idx,
        fpath_attr_type,
        fpath_attr_parent_type,
        fpath_attr_headtail,
        threshold=0.5,
        exclude_atts=[]
    ):
        """Initializes evaluator for attribute prediction on VAW dataset.

        Args:
        - fpath_attr2idx: path to attribute class index file.
        - fpath_attr_type: path to attribute type file.
        - fpath_attr_headtail: path to attribute head/mid/tail categorization file.
        - threshold: positive/negative threshold (for Accuracy metric).
        - exclude_atts: any attribute classes to be excluded from evaluation.
        """

         # Read file that maps from id to attribute name.
        with open(fpath_attr2idx, 'r') as f:
            self.attr2idx = json.load(f)
            self.idx2attr = {v:k for k, v in self.attr2idx.items()}

        # Read file that shows metadata of attributes (e.g., "plaid" is pattern).
        with open(fpath_attr_type, 'r') as f:
            self.attribute_type = json.load(f)
        with open(fpath_attr_parent_type, 'r') as f:
            self.attribute_parent_type = json.load(f)

        # Read file that shows whether attribute is head/mid/tail.
        with open(fpath_attr_headtail, 'r') as f:
            self.attribute_head_tail = json.load(f)

        self.n_class = len(self.attr2idx)
        self.exclude_atts = exclude_atts
        self.threshold = threshold

        # Cache metric score for each class.
        self.score = {} # key: i_class -> value: all metrics.
        self.score_topk = {}

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

        all_groups = ['all', 'head', 'medium', 'tail'] + list(self.attribute_parent_type.keys())
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

            # Add to head/medium/tail group.
            imbalance_group = self.get_attr_head_tail(attr)
            groups_overall[imbalance_group].add_class(class_metric)
            groups_per_class[imbalance_group].add_class(class_metric)

            # Add to corresponding attribute group (color, material, shape, etc.).
            attr_type, attr_subtype = self.get_attr_type(attr)
            groups_overall[attr_type].add_class(class_metric)
            groups_per_class[attr_type].add_class(class_metric)

        # Aggregate final scores.
        # For overall, we're interested in F1.
        # For per-class, we're interested in mean AP, mean recall, mean balanced accuracy.
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
            # Select ony the labeled ones.
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

class COCOAttributesEvaluator:
    def __init__(self, model, dataloader,
                 num_labels=204,
                 batch_size=32, name="attributes",
                 train_set=False, verbose=True):
        super().__init__()
        self.model = model
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.name = name
        self.train_set = train_set

        self.dataloader = dataloader

    def evaluate(self):
        # Array of size the same as the number of labels
        ap = np.zeros((self.num_labels,))
        baseline_ap = np.zeros((self.num_labels,))
        size_counter = 0

        print("# of batches: ", len(self.dataloader))

        ground_truth = np.empty((0, self.num_labels))
        predictions = np.empty((0, self.num_labels))

        # Switch to evaluation mode
        self.model.eval()

        with tqdm(total=len(self.dataloader)) as progress_bar:
            for i, (inp, target) in enumerate(self.dataloader):
                progress_bar.update(1)

                # Load CPU version of target as numpy array
                gts = target.numpy()

                input_var = inp.cuda()
                # compute output
                output = self.model(input_var)
                ests = torch.sigmoid(output).data.cpu().numpy()

                predictions = np.vstack((predictions, ests))
                ground_truth = np.vstack((ground_truth, gts))

                size_counter += ests.shape[0]

        for dim in range(self.num_labels):
            # rescale ground truth to [-1, 1]
            gt = 2 * ground_truth[:, dim] - 1
            est = predictions[:, dim]
            est_base = np.zeros(est.shape)

            ap_score = average_precision(gt, est)
            base_ap_score = average_precision(gt, est_base)
            ap[dim] = ap_score
            baseline_ap[dim] = base_ap_score
        
        # 每类AP
        # for i in range(self.num_labels):
        #     print(ap[i])

        print('*** mAP and Baseline AP scores ***')
        print(np.mean([a if not np.isnan(a) else 0 for a in ap]))
        print(np.mean([a if not np.isnan(a) else 0 for a in baseline_ap]))
