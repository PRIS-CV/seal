from typing import Tuple
from torch import Tensor
import numpy as np

from . import metric
from .metric import Metric


def compute_constrain_violations(adjacency_matrix: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    positive_probabilities = scores
    batch, num_labels = positive_probabilities.shape
    adj = np.broadcast_to(
        adjacency_matrix[None, :, :], (batch, num_labels, num_labels)
    )  # (batch, num_labels, num_labels)
    positive_probabilities_diag = positive_probabilities[
        :, :, None
    ]  # (batch, num_labels, 1)
    positive_probabilities_stack = np.broadcast_to(
        positive_probabilities[:, None, :], (batch, num_labels, num_labels)
    )  # (batch, num_labels, num_labels)

    denominator = np.sum(adj)
    numerator = np.sum(
        adj * (positive_probabilities_stack > positive_probabilities_diag)
    )  # violations
    return numerator, denominator


@metric("ConstraintViolation")
class ConstraintViolation(Metric):
    """
    Given a hierarchy in the form of an adjacency matrix or cooccurence
    statistic in the adjacency matrix format, compute the average
    constraint violation.
    """

    def __init__(self, name, adjacency_matrix: np.ndarray) -> None:
        super().__init__(name)

        self.numerator: float = 0.0
        self.denominator: float = 0.0
        self.adjacency_matrix: np.ndarray = adjacency_matrix
        self.reset()

    def update(self, preds, **kwargs) -> None:

        if isinstance(preds, Tensor):
            outputs = preds.cpu().numpy()
        elif isinstance(preds, np.ndarray):
            outputs = preds
        else:
            raise ValueError("preds should be type of ndarray or tensor")

        numerator, denominator = compute_constrain_violations(self.adjacency_matrix, outputs)
        self.numerator += numerator
        self.denominator += denominator

    def calculate_metric(self):
        assert self.denominator > 0
        metric = self.numerator / self.denominator
        self._result.update({"ConstraintViolation": float(metric)})

    def reset(self) -> None:
        self._result = {}
        self.numerator = 0.0
        self.denominator = 0.0
        