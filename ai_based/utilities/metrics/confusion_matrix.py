from typing import Any, Iterable
from abc import ABC, abstractmethod

import torch

from .base import BaseMetric
from util.datasets import GroundTruthClass


class ConfusionMatrixMetric(BaseMetric):
    """
    Determines the confusion matrix in the form of a Tensor.
    - Dimension one:  True labels
    - Dimension two:  Predicted labels
    """
    def get_name(self) -> str:
        return "confusion_matrix"

    def __call__(self, model_output_batch: torch.Tensor, ground_truth_batch: torch.Tensor) -> torch.Tensor:
        model_output_batch = model_output_batch.cpu()
        ground_truth_batch = ground_truth_batch.cpu()
        predictions = self._compute_class_predictions(model_output_batch)
        stacked = torch.stack((ground_truth_batch, predictions), dim=1)

        batch_confusion_matrix = torch.zeros(size=(len(GroundTruthClass), len(GroundTruthClass))).type(torch.int)
        for gt_, pred_ in stacked:  # TODO Accelerate!
            for g_, p_ in torch.stack((gt_, pred_), dim=1):
                batch_confusion_matrix[g_, p_] += 1
        return batch_confusion_matrix  # FIRST DIM: true labels,  SECOND DIM: predicted labels

    def aggregate_batch_results(self, many_batch_results: Iterable[torch.Tensor]) -> torch.Tensor:
        if isinstance(many_batch_results, torch.Tensor) and many_batch_results.dim() == 2:
            # We got just a single batch. Let's simply return it
            return many_batch_results
        else:
            confusion_matrix = torch.zeros(size=(len(GroundTruthClass), len(GroundTruthClass))).type(torch.int)
            for batch_result in many_batch_results:
                confusion_matrix = confusion_matrix + batch_result
            return confusion_matrix
