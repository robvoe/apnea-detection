from typing import Any, Iterable
from abc import ABC, abstractmethod

import torch

from .base import AveragableMetric


class AccuracyMetric(AveragableMetric):
    def get_name(self) -> str:
        return "accuracy"

    def __call__(self, model_output_batch: torch.Tensor, ground_truth_batch: torch.Tensor) -> Any:
        predictions = self._compute_class_predictions(model_output_batch)
        n_correctly_predicted = torch.eq(predictions, ground_truth_batch).sum()
        return n_correctly_predicted / model_output_batch.shape[0]
