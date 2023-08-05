import numpy as np


def average_precision(truth, scores):
    precision, recall = precision_recall(truth, scores)
    ap = voc_ap(recall, precision)
    return ap


def voc_ap(recall, precision):
    """Pascal VOC AP implementation in pure python"""
    mrec = np.hstack((0, recall, 1))
    mpre = np.hstack((0, precision, 0))

    for i in range(mpre.size-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    i = np.ravel(np.where(mrec[1:] != mrec[0:-1])) + 1

    ap = np.sum((mrec[i]-mrec[i-1]) * mpre[i])
    return ap


def precision_recall(truth, scores):
    """
    Computer precision-recall curve
    :param truth: the ground truth labels. 1 is positive and 0 is negative label
    :param scores: output confidences from the classifier. Values greater than 0.0 are considered positive detections.
    :return:
    """
    sort_inds = np.argsort(-scores, kind='stable')

    # tp = np.cumsum(truth[sort_inds] == 1)
    # fp = np.cumsum(truth[sort_inds] == -1)
    tp = (truth == 1)[sort_inds]
    fp = (truth == 0)[sort_inds]

    tp = np.cumsum(tp.astype(np.float))
    fp = np.cumsum(fp.astype(np.float))

    npos = (truth == 1).sum()

    recall = tp / npos
    precision = tp / (tp + fp)

    return precision, recall


def hamming_distance(gt, pred):
    """
    Get the percentage of correct positive/negative classifications per attribute.
    :param gt:
    :param pred:
    :return:
    """
    return sum([1 for (g, p) in zip(gt, pred) if g == p]) / float(len(gt))


class GroupClassMetric(object):
    def __init__(self, metric_type):
        """This class computes all metrics for a group of attributes.

        Args:
        - metric_type: 'overall' or 'per-class'.
        """
        self.metric_type = metric_type

        if metric_type == 'overall':
            # Keep track of all stats.
            self.true_pos = 0
            self.false_pos = 0
            self.true_neg = 0
            self.false_neg = 0
            self.n_pos = 0
            self.n_neg = 0
        else:
            self.metric = {
                name: []
                for name in ['recall', 'tnr', 'acc', 'bacc', 'precision', 'f1', 'ap']
            }

    def add_class(self, class_metric):
        """Adds computed metrics of a class into this group.
        """
        if self.metric_type == 'overall':
            self.true_pos += class_metric.true_pos
            self.false_pos += class_metric.false_pos
            self.true_neg += class_metric.true_neg
            self.false_neg += class_metric.false_neg
            self.n_pos += class_metric.n_pos
            self.n_neg += class_metric.n_neg
        else:
            self.metric['recall'].append(class_metric.get_recall())
            self.metric['tnr'].append(class_metric.get_tnr())
            self.metric['acc'].append(class_metric.get_acc())
            self.metric['bacc'].append(class_metric.get_bacc())
            self.metric['precision'].append(class_metric.get_precision())
            self.metric['f1'].append(class_metric.get_f1())
            self.metric['ap'].append(class_metric.ap)

    def get_recall(self):
        """Computes recall.
        """
        if self.metric_type == 'overall':
            n_pos_pred = self.true_pos + self.false_pos
            if n_pos_pred == 0:
                # Model makes 0 positive prediction.
                # This is a special case: we fall back to precision = 1 and recall = 0.
                return 0

            if self.n_pos > 0:
                return self.true_pos / self.n_pos
            return -1
        else:
            if -1 not in self.metric['recall']:
                return np.mean(self.metric['recall'])
            return -1

    def get_tnr(self):
        """Computes true negative rate.
        """
        if self.metric_type == 'overall':
            if self.n_neg > 0:
                return self.true_neg / self.n_neg
            return -1
        else:
            if -1 not in self.metric['tnr']:
                return np.mean(self.metric['tnr'])
            return -1

    def get_acc(self):
        """Computes accuracy.
        """
        if self.metric_type == 'overall':
            if self.n_pos + self.n_neg > 0:
                return (self.true_pos + self.true_neg) / (self.n_pos + self.n_neg)
            return -1
        else:
            if -1 not in self.metric['acc']:
                return np.mean(self.metric['acc'])
            return -1

    def get_bacc(self):
        """Computes balanced accuracy.
        """
        if self.metric_type == 'overall':
            recall = self.get_recall()
            tnr = self.get_tnr()
            if recall == -1 or tnr == -1:
                return -1
            return (recall + tnr) / 2.0
        else:
            if -1 not in self.metric['bacc']:
                return np.mean(self.metric['bacc'])
            return -1

    def get_precision(self):
        """Computes precision.
        """
        if self.metric_type == 'overall':
            n_pos_pred = self.true_pos + self.false_pos
            if n_pos_pred == 0:
                # Model makes 0 positive prediction.
                # This is a special case: we fall back to precision = 1 and recall = 0.
                return 1
            return self.true_pos / n_pos_pred
        else:
            if -1 not in self.metric['precision']:
                return np.mean(self.metric['precision'])
            return -1

    def get_f1(self):
        """Computes F1.
        """
        if self.metric_type == 'overall':
            recall = self.get_recall()
            precision = self.get_precision()
            if precision + recall > 0:
                return 2 * precision * recall / (precision + recall)
            return 0
        else:
            if -1 not in self.metric['f1']:
                return np.mean(self.metric['f1'])
            return -1

    def get_ap(self):
        """Computes mAP.
        """
        assert self.metric_type == 'per-class'
        return np.mean(self.metric['ap'])


class SingleClassMetric(object):
    def __init__(self, pred, gt_label):
        """This class computes all metrics for a single attribute.

        Args:
        - pred: np.array of shape [n_instance] -> binary prediction.
        - gt_label: np.array of shape [n_instance] -> groundtruth binary label.
        """
        if pred is None or gt_label is None:
            self.true_pos = 0
            self.false_pos = 0
            self.true_neg = 0
            self.false_neg = 0
            self.n_pos = 0
            self.n_neg = 0
            self.ap = -1
            return

        self.true_pos = ((gt_label == 1) & (pred == 1)).sum()
        self.false_pos = ((gt_label == 0) & (pred == 1)).sum()
        self.true_neg = ((gt_label == 0) & (pred == 0)).sum()
        self.false_neg = ((gt_label == 1) & (pred == 0)).sum()

        # Number of groundtruth positives & negatives.
        self.n_pos = self.true_pos + self.false_neg
        self.n_neg = self.false_pos + self.true_neg
        
        # AP score.
        self.ap = -1

    def get_recall(self):
        """Computes recall.
        """
        n_pos_pred = self.true_pos + self.false_pos
        if n_pos_pred == 0:
            # Model makes 0 positive prediction.
            # This is a special case: we fall back to precision = 1 and recall = 0.
            return 0

        if self.n_pos > 0:
            return self.true_pos / self.n_pos
        return -1

    def get_tnr(self):
        """Computes true negative rate.
        """
        if self.n_neg > 0:
            return self.true_neg / self.n_neg
        return -1

    def get_acc(self):
        """Computes accuracy.
        """
        if self.n_pos + self.n_neg > 0:
            return (self.true_pos + self.true_neg) / (self.n_pos + self.n_neg)
        return -1

    def get_bacc(self):
        """Computes balanced accuracy.
        """
        recall = self.get_recall()
        tnr = self.get_tnr()
        if recall == -1 or tnr == -1:
            return -1
        return (recall + tnr) / 2.0

    def get_precision(self):
        """Computes precision.
        """
        n_pos_pred = self.true_pos + self.false_pos
        if n_pos_pred == 0:
            # Model makes 0 positive prediction.
            # This is a special case: we fall back to precision = 1 and recall = 0.
            return 1
        return self.true_pos / n_pos_pred

    def get_f1(self):
        """Computes F1.
        """
        recall = self.get_recall()
        precision = self.get_precision()
        
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0
