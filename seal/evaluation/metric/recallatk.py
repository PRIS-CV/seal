"""Acknowledgement: This code is modified from mmf.
Page: https://mmf.sh/
Code: https://github.com/facebookresearch/mmf
"""


import warnings
import torch

from .metric import Metric


class RecallAtK(Metric):
    
    def __init__(self, name="recall@k"):
        super().__init__(name)

    def score_to_ranks(self, scores):
        # sort in descending order - largest score gets highest rank
        sorted_ranks, ranked_idx = scores.sort(1, descending=True)

        # convert from ranked_idx to ranks
        ranks = ranked_idx.clone().fill_(0)
        for i in range(ranked_idx.size(0)):
            for j in range(100):
                ranks[i][ranked_idx[i][j]] = j
        ranks += 1
        return ranks

    def get_gt_ranks(self, ranks, ans_ind):
        _, ans_ind = ans_ind.max(dim=1)
        ans_ind = ans_ind.view(-1)
        gt_ranks = torch.LongTensor(ans_ind.size(0))

        for i in range(ans_ind.size(0)):
            gt_ranks[i] = int(ranks[i, ans_ind[i].long()])
        return gt_ranks

    def process_ranks(self, ranks):
        num_opts = 100

        # none of the values should be 0, there is gt in options
        if torch.sum(ranks.le(0)) > 0:
            num_zero = torch.sum(ranks.le(0))
            warnings.warn(f"Some of ranks are zero: {num_zero}")
            ranks = ranks[ranks.gt(0)]

        # rank should not exceed the number of options
        if torch.sum(ranks.ge(num_opts + 1)) > 0:
            num_ge = torch.sum(ranks.ge(num_opts + 1))
            warnings.warn(f"Some of ranks > 100: {num_ge}")
            ranks = ranks[ranks.le(num_opts + 1)]
        return ranks

    def get_ranks(self, sample_list, model_output, *args, **kwargs):
        output = model_output["scores"]
        expected = sample_list["targets"]

        ranks = self.score_to_ranks(output)
        gt_ranks = self.get_gt_ranks(ranks, expected)

        ranks = self.process_ranks(gt_ranks)
        return ranks.float()

    def calculate(self, sample_list, model_output, k, *args, **kwargs):
        ranks = self.get_ranks(sample_list, model_output)
        recall = float(torch.sum(torch.le(ranks, k))) / ranks.size(0)
        return recall
