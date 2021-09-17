from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any, Tuple, SupportsFloat, List

import torch

from .metrics.base import BaseMetric
from .metrics.accuracy import AccuracyMetric
from .metrics.confusion_matrix import ConfusionMatrixMetric
from util.datasets import GroundTruthClass


class BaseEvaluator(ABC):
    """
    Evaluator classes wrap the necessary logic to test the performance of a model. Subclasses implementations depend on
    the model output domain and the ground truth domain.
    """
    @abstractmethod
    def get_all_metrics(self) -> Iterable[BaseMetric]:
        """
        :return: Tuple containing all the internally used metric objects.
        """
        pass

    def __call__(self, model_output_batch: torch.Tensor, ground_truth_batch: torch.Tensor) -> Dict[str, Any]:
        """
        Computes a batch-based evaluation result, using all available metrics.

        @param model_output_batch: Batch of model output tensors.
        @param ground_truth_batch: Batch of corresponding ground truth values.
        @return: A dictionary that maps metric-names to their respective results.
        """
        return {metric.get_name(): metric(model_output_batch, ground_truth_batch) for metric in self.get_all_metrics()}

    def aggregate_batch_results(self, many_batch_results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates multiple batch results, previously obtained with the __call__ function.
        
        :param many_batch_results: Iterable of result dictionaries that should be aggregated
        :return: Aggregated result dictionary (same structure as the input sub-dictionaries)
        """
        aggregated_results = {}
        for metric in self.get_all_metrics():
            metric_name = metric.get_name()
            metric_results = [result[metric_name] for result in many_batch_results]
            aggregated_results[metric_name] = metric.aggregate_batch_results(metric_results)
        return aggregated_results

    def print_exhausting_metrics_results(self, aggregated_results: Dict[str, Any], indent=0, flat=False) -> None:
        """
        Format and print the evaluation results nicely.
        :param aggregated_results: dictionary containing all evaluation results
        :param indent: indent everything by a number of tabs
        :param flat: print everything in one line if set to `True`
        """
        def _print_inner(aggregated_results_: Dict, indent_: int):
            for metric_name_, result_ in aggregated_results_.items():
                if isinstance(result_, torch.Tensor) and result_.dim() == 1:
                    tensor_str = "   ".join([f"{v_:.4f}" for v_ in result_])
                    print("\t" * indent_ + f"{metric_name_}: {tensor_str}")
                elif isinstance(result_, torch.Tensor) and result_.dim() != 0:
                    print("\t" * indent_ + f"{metric_name_}: {result_}")
                elif isinstance(result_, dict):
                    print("\t" * indent_ + f"{metric_name_}:")
                    _print_inner(result_, indent_=indent_+1)
                else:
                    print("\t" * indent_ + f"{metric_name_}: {result_:.4f}")

        if flat:
            s = "\t" * indent
            s += self.get_short_performance_summary(aggregated_results) + " -- "
            for metric_name, result in aggregated_results.items():
                s += f"{metric_name}: {result} | "
            print(s)
        else:
            print("\t" * indent + f"Performance summary: {self.get_short_performance_summary(aggregated_results)}")
            _print_inner(aggregated_results, indent_=indent)

    @abstractmethod
    def get_short_performance_summary(self, aggregated_results: Dict[str, Any]) -> str:
        """
        :return: Short, printable summary of the model performance for easy direct comparison.
        """
        pass

    @abstractmethod
    def is_better_than(self, aggregated_results_a: Dict[str, Any], aggregated_results_b: Dict[str, Any]) -> bool:
        """
        Compares two dictionaries with evaluation results following certain criteria.
        :return: `True` if a is better than b, otherwise `False`.
        """
        pass


class AccuracyEvaluator(BaseEvaluator):
    def __init__(self):
        self._accuracy_metric = AccuracyMetric()

    def get_all_metrics(self) -> Iterable[BaseMetric]:
        return [self._accuracy_metric]

    def get_short_performance_summary(self, aggregated_results: Dict[str, Any]) -> str:
        return f"Accuracy: {aggregated_results[self._accuracy_metric.get_name()]*100:.2f}%"

    def is_better_than(self, aggregated_results_a: Dict[str, Any], aggregated_results_b: Dict[str, Any]) -> bool:
        error_metric_name = self._accuracy_metric.get_name()
        return aggregated_results_a[error_metric_name] > aggregated_results_b[error_metric_name]


class ConfusionMatrixEvaluator(BaseEvaluator):
    def __init__(self):
        self._confusion_matrix_metric = ConfusionMatrixMetric()

    def get_all_metrics(self) -> Iterable[BaseMetric]:
        return [self._confusion_matrix_metric]

    @staticmethod
    def _safe_mean(tensor_: torch.Tensor) -> torch.Tensor:
        tensor_[torch.isnan(tensor_)] = 0.0
        return tensor_.mean()

    def aggregate_batch_results(self, many_batch_results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        aggregated_results = super(ConfusionMatrixEvaluator, self).aggregate_batch_results(many_batch_results)
        aggregated_confusion_matrix = aggregated_results[self._confusion_matrix_metric.get_name()]

        class_occurrences = torch.sum(aggregated_confusion_matrix, dim=1)
        class_weights = class_occurrences / class_occurrences.sum()

        # Handle precision, recall and f1-score
        class_based_precision, class_based_recall = self._get_class_based_precision_recall(aggregated_confusion_matrix)

        macro_precision = self._safe_mean(class_based_precision)  # class_based_precision.mean()
        macro_recall = self._safe_mean(class_based_recall)  # class_based_recall.mean()
        weighted_precision = (class_based_precision * class_weights).sum()
        weighted_recall = (class_based_recall * class_weights).sum()
        macro_f1_score = (macro_precision * macro_recall * 2) / (macro_precision + macro_recall)
        weighted_f1_score = (weighted_precision * weighted_recall * 2) / (weighted_precision + weighted_recall)

        return {
            self._confusion_matrix_metric.get_name(): {
                "class_based_precision": class_based_precision,
                "class_based_recall": class_based_recall,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
                "macro_f1_score": macro_f1_score,
                "weighted_f1_score": weighted_f1_score
            }
        }

    @staticmethod
    def _get_class_based_precision_recall(aggregated_confusion_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determines precision and recall for each single class."""
        precision = torch.zeros(size=(len(GroundTruthClass),)).type(torch.float)
        recall = torch.zeros(size=(len(GroundTruthClass),)).type(torch.float)
        for klass in range(len(GroundTruthClass)):
            p_ = aggregated_confusion_matrix[klass, klass] / aggregated_confusion_matrix[:, klass].sum()  # can be NaN!
            r_ = aggregated_confusion_matrix[klass, klass] / aggregated_confusion_matrix[klass, :].sum()  # can be NaN!
            precision[klass] = p_
            recall[klass] = r_
        return precision, recall

    def get_short_performance_summary(self, aggregated_results: Dict[str, Any]) -> str:
        intent = " " * 10
        macro_precision = aggregated_results[self._confusion_matrix_metric.get_name()]["macro_precision"]
        macro_recall = aggregated_results[self._confusion_matrix_metric.get_name()]["macro_recall"]
        macro_f1_score = aggregated_results[self._confusion_matrix_metric.get_name()]["macro_f1_score"]
        return f"\n" \
               f"{intent}Macro precision: {macro_precision:.3f}" \
               f"{intent}Macro recall: {macro_recall:.3f}" \
               f"{intent}Macro f1-score: {macro_f1_score:.3f}"

    def is_better_than(self, aggregated_results_a: Dict[str, Any], aggregated_results_b: Dict[str, Any]) -> bool:
        macro_f1_score_a = aggregated_results_a[self._confusion_matrix_metric.get_name()]["macro_f1_score"]
        macro_f1_score_b = aggregated_results_b[self._confusion_matrix_metric.get_name()]["macro_f1_score"]
        if torch.isnan(macro_f1_score_b) and not torch.isnan(macro_f1_score_a):
            return True
        return macro_f1_score_a >= macro_f1_score_b
