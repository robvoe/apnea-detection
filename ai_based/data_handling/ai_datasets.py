import abc
import functools
from collections import Counter
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pathlib
import pickle
from typing import Dict, Tuple, List, Optional, Union, Iterable, NamedTuple
from datetime import datetime
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch.utils.data
import torch
import pandas as pd

from util.datasets.sliding_window import GroundTruthClass, SlidingWindowDataset
from util.mathutil import normalize_robust


SlidingWindowTimestamps = NamedTuple("SlidingWindowTimestamps", center_point=pd.Timedelta, features=pd.TimedeltaIndex, ground_truth=pd.TimedeltaIndex)

FEATURE_SIGNAL_NAMES = ("SaO2", "ABD", "CHEST", "AIRFLOW")  # The signals that end up in the outputted feature map


class BaseAiDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_gt_class_occurrences(self) -> Dict[GroundTruthClass, int]:
        pass


class AiDataset(BaseAiDataset):
    """
    Loads one or multiple SlidingWindowDatasets, unifies them and
    - makes their sliding windows accessible via a single index
    - transforms the sliding window data (features & ground truth), such that it can be applied for AI trainings and
      inference
    - applies preprocessing steps (normalizing etc.) to the features before outputting
    """
    @dataclass
    class Config:
        sliding_window_dataset_config: SlidingWindowDataset.Config
        dataset_folders: List[Path]
        noise_mean_std: Optional[Tuple[float, float]]
        normalize_signals: List[str] = ("AIRFLOW", "ABD", "CHEST")  # Denotes which of the signals should be normalized

        def __post_init__(self):
            assert all(s in FEATURE_SIGNAL_NAMES for s in self.normalize_signals), \
                f"At least one of the given 'normalize_signals' is not contained in {FEATURE_SIGNAL_NAMES}!"

    def __init__(self, config: Config):
        super(AiDataset, self).__init__(config=config)

        # Let's load the underlying SlidingWindowDatasets and check if all signals are provided
        self._sliding_window_datasets = []
        for dataset_folder in self.config.dataset_folders:
            sliding_window_dataset = SlidingWindowDataset(config=config.sliding_window_dataset_config,
                                                          dataset_folder=dataset_folder, allow_caching=True)
            assert all([s in sliding_window_dataset.signals for s in FEATURE_SIGNAL_NAMES]), \
                f"{SlidingWindowDataset.__name__} '{sliding_window_dataset.config.dataset_folder.name}' " \
                f"does not provide all necessary signal names {FEATURE_SIGNAL_NAMES}; at least one signal is missing!"
            self._sliding_window_datasets += [sliding_window_dataset]

        # Now, let's take a look at the dataset lengths
        self._sliding_window_datasets_lengths = [len(ds) for ds in self._sliding_window_datasets]
        self._len = sum(self._sliding_window_datasets_lengths)

        # Let's build-up the cache for function _resolve_index() for shorter execution times
        _ = [self._resolve_index(i) for i in range(self._len)]
        del _

        # Pre-JIT our normalization method to save time when querying data in multi-process context
        normalize_robust(np.zeros(shape=(5,)))

    def get_sliding_window_timestamps(self, idx) -> SlidingWindowTimestamps:
        """Returns all timestamps that belong to the sliding windows (features & gt) obtained via __getitem__"""
        assert -len(self) <= idx < len(self), "Index out of bounds"
        if idx < 0:
            idx = idx + len(self)
        dataset_index, dataset_internal_index = self._resolve_index(idx)
        window_data = self._sliding_window_datasets[dataset_index][dataset_internal_index]

        features_index, gt_index = window_data.signals.index, window_data.ground_truth.index
        assert isinstance(features_index, pd.TimedeltaIndex) and isinstance(gt_index, pd.TimedeltaIndex)

        return SlidingWindowTimestamps(center_point=window_data.center_point, features=features_index, ground_truth=gt_index)

    @functools.cache
    def _resolve_index(self, idx: int) -> Tuple[int, int]:
        """Resolves a given index to sliding-window-dataset & dataset-internal index."""
        assert 0 <= idx < len(self), "Index out of bounds"
        accumulated_dataset_lengths = 0
        for dataset_index, dataset_length in enumerate(self._sliding_window_datasets_lengths):
            previous_ = accumulated_dataset_lengths
            accumulated_dataset_lengths += dataset_length
            if previous_ <= idx < accumulated_dataset_lengths:
                dataset_internal_index = idx - previous_
                return dataset_index, dataset_internal_index
        raise RuntimeError("We shouldn't have ended up here!")

    def __getitem__(self, idx: int):
        assert -len(self) <= idx < len(self), "Index out of bounds"
        if idx < 0:
            idx = idx + len(self)
        dataset_index, dataset_internal_index = self._resolve_index(idx)
        window_data = self._sliding_window_datasets[dataset_index][dataset_internal_index]

        features = np.zeros(shape=(len(FEATURE_SIGNAL_NAMES), len(window_data.signals)), dtype=np.float)
        for i, signal_name in enumerate(FEATURE_SIGNAL_NAMES):
            raw_signal_data = window_data.signals[signal_name].values
            data = raw_signal_data if signal_name not in self.config.normalize_signals else normalize_robust(raw_signal_data)
            features[i, :] = data

        gt = np.array([gt.value for gt in window_data.ground_truth], dtype=np.int)

        # Convert samples into tensors
        features_tensor = torch.from_numpy(features).type(torch.float)
        gt_tensor = torch.tensor(gt, dtype=torch.long)

        # Add noise
        if self.config.noise_mean_std is not None:
            noise = torch.normal(mean=self.config.noise_mean_std[0], std=self.config.noise_mean_std[1],
                                 size=features_tensor.size())
            features_tensor += noise

        return features_tensor, gt_tensor

    def __len__(self):
        return self._len

    def get_gt_class_occurrences(self) -> Dict[GroundTruthClass, int]:
        gt_class_occurrences_sum: Dict[GroundTruthClass, int] = {klass: 0 for klass in GroundTruthClass}
        for dataset in self._sliding_window_datasets:
            occ_ = dataset.gt_class_occurrences
            for k, v in occ_.items():
                gt_class_occurrences_sum[k] += v
        return gt_class_occurrences_sum


@pytest.fixture
def ai_dataset_provider() -> AiDataset:
    from util.paths import DATA_PATH

    config = AiDataset.Config(
        sliding_window_dataset_config=SlidingWindowDataset.Config(
            downsample_frequency_hz=5,
            time_window_size=pd.Timedelta("5 minutes"),
            time_window_stride=11,
            ground_truth_vector_width=11
        ),
        dataset_folders=[DATA_PATH / "training" / "tr07-0168"],
        noise_mean_std=(0, 0.5),
    )
    ai_dataset = AiDataset(config=config)
    print()
    print()
    print(f"len(dataset) = {len(ai_dataset):,}")
    return ai_dataset


def test_development(ai_dataset_provider):
    ai_dataset = ai_dataset_provider
    len_ = len(ai_dataset)
    ground_truth_occurrences = ai_dataset.get_gt_class_occurrences()

    features, gt = ai_dataset[0]
    pass


def test_performance(ai_dataset_provider):
    from datetime import datetime

    n_cycles = 1

    ai_dataset = ai_dataset_provider
    started_at = datetime.now()
    for c in range(n_cycles):
        for i in range(len(ai_dataset)):
            features, gt = ai_dataset[i]
    duration_seconds = (datetime.now()-started_at).total_seconds()

    print()
    print()
    print(f"Overall execution time: {duration_seconds:.1f}s")
    print()
    print(f"Total duration per whole dataset cycle: {duration_seconds/n_cycles:.1f}s")
    print(f"Duration per index read: {duration_seconds/n_cycles/len(ai_dataset)*1000:.2f}ms")
