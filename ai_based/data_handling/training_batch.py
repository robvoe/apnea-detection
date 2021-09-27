from abc import ABC, abstractmethod
from typing import Sized, List

import torch


class BaseTrainingBatch(ABC):
    """
    A data structure that encapsules all the relevant data for a single batch for training a neural network.
    """
    def __init__(self, input_data: torch.Tensor, ground_truth: Sized, sample_indexes: List[int]):
        # The input data must be presented as a tensor already (intended for direct use with a network)
        if not isinstance(input_data, torch.Tensor):
            raise TypeError(f"Input data must be a torch tensor but got type: {type(input_data)}")
        assert len(input_data) == len(ground_truth) == len(sample_indexes), \
            f"Lengths of input/gt/sample-indexes mismatching: {len(input_data)} vs {len(ground_truth)} vs {len(sample_indexes)}"

        self.input_data = input_data
        self.ground_truth = ground_truth  # gt can be of any data format but must be an iterable with the same batch size as the input data.
        self.sample_indexes = sample_indexes

        self.device = input_data.device

    def __len__(self):
        return self.input_data.shape[0]

    @classmethod
    @abstractmethod
    def from_iterable(cls, samples, device="cpu"):
        """
        Factory function that constructs a batch from an iterable of data samples.
        """
        pass

    @abstractmethod
    def to_device(self, device):
        """
        Copy all contained tensors to the target device.
        """
        pass


class TrainingBatch(BaseTrainingBatch):
    def __init__(self, feature_matrixes: torch.Tensor, ground_truth_matrixes: torch.Tensor, sample_indexes: List[int]):
        assert feature_matrixes.dim() == 3
        assert ground_truth_matrixes.dim() == 2
        super().__init__(feature_matrixes, ground_truth_matrixes, sample_indexes=sample_indexes)

    @classmethod
    def from_iterable(cls, samples, device="cpu"):
        feature_matrixes = []
        ground_truth_matrixes = []
        sample_indexes = []
        for features, ground_truth, idx in samples:
            feature_matrixes.append(features)
            ground_truth_matrixes.append(ground_truth)
            sample_indexes.append(idx)
        return cls(torch.stack(feature_matrixes).to(device), torch.stack(ground_truth_matrixes).to(device), sample_indexes=sample_indexes)

    def to_device(self, device):
        self.input_data = self.input_data.to(device)
        self.ground_truth = self.ground_truth.to(device)
