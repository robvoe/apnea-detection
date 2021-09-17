from abc import ABC, abstractmethod
from typing import Any, Iterable

import torch


class BaseMetric(ABC):
    """
    Base class for all metrics. A metric is a function that compares the output of a model and the corresponding ground
    truth.
    """
    @abstractmethod
    def get_name(self) -> str:
        """Every metric should get a short unique name."""
        pass

    @abstractmethod
    def __call__(self, model_output_batch: torch.Tensor, ground_truth_batch: torch.Tensor) -> Any:
        """
        Computes the metric for a given batch.

        @param model_output_batch: Batch of model output tensors.
        @param ground_truth_batch: Batch of corresponding ground truth values.
        @return: Computed metric value(s) for the given batch.
        """
        pass

    @abstractmethod
    def aggregate_batch_results(self, many_batch_results: Iterable[Any]) -> Any:
        """
        Aggregates multiple batch results, previously obtained with the __call__ function.

        :param many_batch_results: Iterable of multiple results, previously derived using the __call__ function.
        :return: A meaningful metric, aggregated on multiple batches.
        """
        pass

    @staticmethod
    def _compute_class_predictions(model_output_batch: torch.Tensor) -> torch.Tensor:
        return torch.argmax(model_output_batch, dim=1)


class AveragableMetric(BaseMetric, ABC):
    def aggregate_batch_results(self, many_batch_results):
        return torch.tensor(many_batch_results, device=many_batch_results[0].device).mean()
