from abc import ABC, abstractmethod
from typing import Sized

import torch


class BaseTrainingBatch(ABC):
    """
    A data structure that encapsules all the relevant data for a single batch for training a neural network.
    """
    def __init__(self, input_data: torch.Tensor, ground_truth: Sized):
        # The input data must be presented as a tensor already (intended for direct use with a network)
        if not isinstance(input_data, torch.Tensor):
            raise TypeError(f"Input data must be a torch tensor but got type: {type(input_data)}")
        self.input_data = input_data

        # The ground truth can be of any data format but must be an iterable with the same batch size as the input data.
        if len(input_data) != len(ground_truth):
            raise ValueError(
                f"Batch size for input and ground truth doesn't match: {len(input_data)} vs {len(ground_truth)}")
        self.ground_truth = ground_truth

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
    def __init__(self, feature_matrices: torch.Tensor, ground_truth_matrices: torch.Tensor):
        assert feature_matrices.dim() == 3
        assert ground_truth_matrices.dim() == 2
        super().__init__(feature_matrices, ground_truth_matrices)

    @classmethod
    def from_iterable(cls, samples, device="cpu"):
        feature_matrices = []
        ground_truth_matrices = []
        for features, ground_truth in samples:
            feature_matrices.append(features)
            ground_truth_matrices.append(ground_truth)
        return cls(torch.stack(feature_matrices).to(device), torch.stack(ground_truth_matrices).to(device))

    def to_device(self, device):
        self.input_data = self.input_data.to(device)
        self.ground_truth = self.ground_truth.to(device)
