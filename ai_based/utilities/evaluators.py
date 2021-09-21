import functools
from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any, Tuple, SupportsFloat, List, Optional, Union

import torch
import numpy as np
import numba

from .metrics.base import BaseMetric
from .metrics.accuracy import AccuracyMetric
from .metrics.confusion_matrix import ConfusionMatrixMetric
from util.datasets import GroundTruthClass


class BaseEvaluator(ABC):
    @abstractmethod
    def __init__(self, model_output_batch: Optional[torch.Tensor], ground_truth_batch: Optional[torch.Tensor]):
        """Constructs an Evaluator instance. All parameters should either contain values or be None."""
        assert (model_output_batch is None and ground_truth_batch is None) or \
               (model_output_batch is not None and ground_truth_batch is not None), \
               "Either both of the parameters 'model_output_batch' and 'ground_truth_batch' should hold values or be None!"

    @classmethod
    def empty(cls) -> "BaseEvaluator":
        """Constructs an empty Evaluator instance which, so far, stores no metrics."""
        return cls(model_output_batch=None, ground_truth_batch=None)

    @abstractmethod
    def __add__(self, other: "BaseEvaluator") -> "BaseEvaluator":
        """Adds two Evaluator instances. The intention is to unify multiple batches into one Evaluator."""
        pass

    @abstractmethod
    def get_short_summary(self) -> str:
        """Outputs a short summary on the so-far stored metrics."""
        pass

    @abstractmethod
    def _get_comparable_score(self) -> np.float:
        """Creates a single score that can be used for performance-comparison purposes."""
        pass

    @staticmethod
    def _safe_mean(vector_: Union[torch.Tensor, np.ndarray]) -> Union[np.ndarray, torch.Tensor]:
        """Helper function that, prior to arithmetic-mean determination, sets NaN values to zero."""
        if isinstance(vector_, torch.Tensor):
            vector_[torch.isnan(vector_)] = 0.0
        else:
            vector_[np.isnan(vector_)] = 0.0
        return vector_.mean()

    @staticmethod
    def _compute_class_predictions(model_output_batch: torch.Tensor) -> torch.Tensor:
        return torch.argmax(model_output_batch, dim=1)

    @abstractmethod
    def get_scores_dict(self) -> Dict[str, np.ndarray]:
        """Returns a dictionary with all relevant scores calculated."""
        pass

    def __gt__(self, other: "BaseEvaluator") -> bool:
        """Determines which of the two Evaluator instances holds better scores."""
        my_score = self._get_comparable_score()
        other_score = other._get_comparable_score()
        if np.isnan(other_score) and not np.isnan(my_score):
            return True
        return my_score > other_score

    def print_exhausting_metrics_results(self, indent: int = 0, flat: bool = False) -> None:
        """
        Pretty-print the evaluation results.

        :param indent: indent everything by a number of tabs
        :param flat: print everything in one line if set to `True`
        """
        def _print_inner(scores_dict_: Dict, indent_: int):
            for score_, val_ in scores_dict_.items():
                if (isinstance(val_, torch.Tensor) and val_.dim() == 1) or (isinstance(val_, np.ndarray) and val_.ndim == 1):
                    vector_str = "   ".join([f"{v_:.4f}" for v_ in val_])
                    print("\t" * indent_ + f"{score_}: {vector_str}")
                elif (isinstance(val_, torch.Tensor) and val_.dim() != 0) or (isinstance(val_, np.ndarray) and val_.ndim != 0):
                    print("\t" * indent_ + f"{score_}: {val_}")
                elif isinstance(val_, dict):
                    print("\t" * indent_ + f"{score_}:")
                    _print_inner(val_, indent_=indent_+1)
                else:
                    print("\t" * indent_ + f"{score_}: {val_:.4f}")

        if flat:
            s = "\t" * indent
            s += self.get_short_summary() + " -- "
            for score_name, result in self.get_scores_dict().items():
                s += f"{score_name}: {result} | "
            print(s)
        else:
            print("\t" * indent + f"Short summary: {self.get_short_summary()}")
            _print_inner(self.get_scores_dict(), indent_=indent)


class ConfusionMatrixEvaluator(BaseEvaluator):
    def __init__(self, model_output_batch: Optional[torch.Tensor], ground_truth_batch: Optional[torch.Tensor]):
        super(ConfusionMatrixEvaluator, self).__init__(model_output_batch=model_output_batch, ground_truth_batch=ground_truth_batch)

        self.__confusion_matrix: np.array = np.zeros(shape=(len(GroundTruthClass), len(GroundTruthClass)), dtype=int)
        if model_output_batch is None or ground_truth_batch is None:
            return

        model_output_batch = model_output_batch.cpu()
        ground_truth_batch = ground_truth_batch.cpu()
        predictions = self._compute_class_predictions(model_output_batch)

        self.__fill_confusion_matrix(confusion_matrix=self.__confusion_matrix,
                                     ground_truth_samples=ground_truth_batch.view(-1).numpy(),
                                     predicted_samples=predictions.view(-1).numpy())

    @staticmethod
    @numba.jit(nopython=True)
    def __fill_confusion_matrix(confusion_matrix: np.ndarray, ground_truth_samples: np.ndarray, predicted_samples: np.ndarray) -> None:
        for gt_, pred_ in zip(ground_truth_samples, predicted_samples):
            confusion_matrix[gt_, pred_] += 1

    def __add__(self, other: "BaseEvaluator") -> "BaseEvaluator":
        assert isinstance(other, ConfusionMatrixEvaluator)
        new_evaluator = ConfusionMatrixEvaluator.empty()
        new_evaluator.__confusion_matrix = self.__confusion_matrix + other.__confusion_matrix
        return new_evaluator

    @functools.cache
    def get_scores_dict(self) -> Dict[str, np.ndarray]:
        class_based_precision, class_based_recall = self.__get_class_based_precision_recall(self.__confusion_matrix)
        macro_precision = self._safe_mean(class_based_precision)
        macro_recall = self._safe_mean(class_based_recall)
        with np.errstate(divide='ignore', invalid='ignore'):  # Helps to suppress divide-by-0-warnings
            macro_f1_score = (macro_precision * macro_recall * 2) / (macro_precision + macro_recall)
            macro_f1_score = np.nan_to_num(macro_f1_score)
        return {
            "class_based_precision": class_based_precision,
            "class_based_recall": class_based_recall,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1_score": macro_f1_score
        }

    def get_short_summary(self, padding: int = 3) -> str:
        scores = self.get_scores_dict()
        padding_str = " " * 10
        return f"Macro f1-score: {scores['macro_f1_score']:.3f}" \
               f"{padding_str}Macro precision: {scores['macro_precision']:.3f}" \
               f"{padding_str}Macro recall: {scores['macro_recall']:.3f}"

    def _get_comparable_score(self) -> np.float:
        scores = self.get_scores_dict()
        return scores["macro_f1_score"]

    @staticmethod
    def __get_class_based_precision_recall(aggregated_confusion_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Determines precision and recall for each single class."""
        precision = np.zeros(shape=(len(GroundTruthClass),), dtype=float)
        recall = np.zeros(shape=(len(GroundTruthClass),), dtype=float)
        for klass in range(len(GroundTruthClass)):
            with np.errstate(divide='ignore', invalid='ignore'):  # Helps to suppress divide-by-0-warnings
                p_ = aggregated_confusion_matrix[klass, klass] / aggregated_confusion_matrix[:, klass].sum()  # can be NaN!
                r_ = aggregated_confusion_matrix[klass, klass] / aggregated_confusion_matrix[klass, :].sum()  # can be NaN!
            precision[klass] = p_
            recall[klass] = r_
        return precision, recall

    def get_confusion_matrix(self) -> np.ndarray:
        return self.__confusion_matrix
