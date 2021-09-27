import abc
import copy
import functools
import os
from collections import Counter
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pathlib
import pickle
from typing import Dict, Tuple, List, Optional, Union, Iterable, NamedTuple, Any
from datetime import datetime
from copy import deepcopy
from pathlib import Path
import multiprocessing as mp

import numba
import numba.typed
import numpy as np
import pytest
import torch.utils.data
import torch
import pandas as pd
from tqdm import tqdm

from util.datasets.sliding_window import GroundTruthClass, SlidingWindowDataset
from util.mathutil import normalize_robust, PeakType
from util.filter import apply_butterworth_lowpass_filter
from util.mathutil import get_peaks


SlidingWindowTimestamps = NamedTuple("SlidingWindowTimestamps", dataset_index=int, center_point=pd.Timedelta, features=pd.TimedeltaIndex, ground_truth=pd.TimedeltaIndex)

FEATURE_SIGNAL_NAMES = ("SaO2", "ABD", "CHEST", "AIRFLOW")  # The signals that end up in the outputted feature map
NORMALIZE_SIGNAL_NAMES = ("ABD", "CHEST", "AIRFLOW")  # Signals that we wish to normalize at window level


class BaseAiDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def __getitem__(self, idx) -> Tuple[torch.Tensor, Any, int]:
        """Returns a Tuple of three elements:  features, ground truth, sample index"""
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
    @staticmethod
    def _load_prepare_dataset(dataset_folder: Path, sliding_window_dataset_config: SlidingWindowDataset.Config) -> SlidingWindowDataset:
        ds_ = SlidingWindowDataset(config=sliding_window_dataset_config, dataset_folder=dataset_folder, allow_caching=True)
        assert all([s in ds_.signals for s in FEATURE_SIGNAL_NAMES]), \
            f"{SlidingWindowDataset.__name__} '{dataset_folder.name}' does not provide all necessary " \
            f"signal names {FEATURE_SIGNAL_NAMES}; at least one out of them is missing!"
        # Reduce signal complexity by determining hull curve of our alternating signals. Therefore we use "get_peak"
        for signal_name in ("ABD", "CHEST", "AIRFLOW"):
            # Pre-filter our signal with, to provide a bit more stable peak-detection
            prefiltered_signal_array = apply_butterworth_lowpass_filter(data=ds_.signals[signal_name].values, f_cutoff=0.1, f_sample=1, filter_order=5)
            filter_kernel_width = int(sliding_window_dataset_config.downsample_frequency_hz * 0.7)
            peaks_list = get_peaks(waveform=prefiltered_signal_array, filter_kernel_width=filter_kernel_width)
            # Create an empty NaN-array and fill in the absolute magnitudes at each peak's center point
            peakified_signal_array = np.zeros(shape=prefiltered_signal_array.shape, dtype=np.float)
            if len(peaks_list) <= 2:
                pass  # This covers the -very unlikely yet present- case that the signal has no peaks (e.g. 07-0709)
            else:
                peakified_signal_array[:] = np.nan
                for p in peaks_list:
                    peakified_signal_array[p.center] = np.abs(p.extreme_value)
            # Now, let's interpolate the NaNs in between the points
            peakified_signal_series = pd.Series(data=peakified_signal_array, index=ds_.signals.index, dtype=np.float32)
            peakified_signal_series = peakified_signal_series.interpolate(method="linear").bfill().ffill()
            ds_.signals[signal_name] = peakified_signal_series
        assert not np.any(np.isnan(ds_.signals.values)), f"Oops, there's a NaN value in dataset '{dataset_folder.name}'"
        assert not np.any(np.isinf(ds_.signals.values)), f"Oops, there's a inf value in dataset '{dataset_folder.name}'"
        return ds_

    @dataclass
    class Config:
        sliding_window_dataset_config: SlidingWindowDataset.Config
        dataset_folders: List[Path]
        noise_mean_std: Optional[Tuple[float, float]]

    def __init__(self, config: Config, progress_message: str = "Loading and pre-processing dataset", n_processes: int = None):
        """
        Creates an AiDataset

        @param config: Config to this AiDataset instance
        @param progress_message: Message that shall be shown during dataset load/preprocessing
        @param n_processes: Number of processes that shall be used for loading. If None, the number is
                            automatically determined.
        """
        super(AiDataset, self).__init__(config=config)

        affinity: int = len(os.sched_getaffinity(0))
        if n_processes is None:
            n_processes = max(1, int(affinity/2))
        assert n_processes >= 1, f"Passed parameter 'n_processes' ({n_processes}) must be None, or at least 1!"
        assert n_processes <= len(os.sched_getaffinity(0)), \
            f"Passed parameter 'n_processes' ({n_processes}) is larger than maximum number of possible processes ({affinity})!"

        # Let's load the underlying SlidingWindowDatasets and check if all signals are provided
        # self._load_prepare_dataset(dataset_folder=self.config.dataset_folders[0], sliding_window_dataset_config=config.sliding_window_dataset_config)
        get_peaks(waveform=np.array([-1, 0, 1, 0]), filter_kernel_width=2)  # Pre-JIT get_peaks to improve performance
        with mp.Pool(processes=n_processes) as pool:
            load_fn_ = functools.partial(self._load_prepare_dataset, sliding_window_dataset_config=config.sliding_window_dataset_config)
            loading_results = list(tqdm(pool.imap(load_fn_, self.config.dataset_folders), desc=progress_message,
                                        total=len(self.config.dataset_folders)))
            self._sliding_window_datasets: List[SlidingWindowDataset] = loading_results

        # Now, let's take a look at the dataset lengths. Here, for speed-up reasons in conjunction with
        # our function '_resolve_index_helper', we make use of Numba lists
        self._sliding_window_datasets_lengths = numba.typed.List()
        [self._sliding_window_datasets_lengths.append(len(ds)) for ds in self._sliding_window_datasets]
        self._len = sum(self._sliding_window_datasets_lengths)

        # Pre-JIT our '_resolve_index_helper' function, for performance improvement
        _ = self._resolve_index_helper(idx=0, sliding_window_dataset_lengths=self._sliding_window_datasets_lengths)

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

        return SlidingWindowTimestamps(dataset_index=dataset_index, center_point=window_data.center_point, features=features_index, ground_truth=gt_index)

    @staticmethod
    @numba.jit(nopython=True)
    def _resolve_index_helper(idx: int, sliding_window_dataset_lengths: numba.typed.List) -> Tuple[int, int]:
        accumulated_dataset_lengths = 0
        for dataset_index, dataset_length in enumerate(sliding_window_dataset_lengths):
            previous_ = accumulated_dataset_lengths
            accumulated_dataset_lengths += dataset_length
            if previous_ <= idx < accumulated_dataset_lengths:
                dataset_internal_index = idx - previous_
                return dataset_index, dataset_internal_index
        raise RuntimeError("We shouldn't have ended up here!")

    def _resolve_index(self, idx: int) -> Tuple[int, int]:
        """Resolves a given index to sliding-window-dataset & dataset-internal index."""
        assert 0 <= idx < len(self), "Index out of bounds"
        return self._resolve_index_helper(idx=idx, sliding_window_dataset_lengths=self._sliding_window_datasets_lengths)

    def __getitem__(self, idx: int):
        assert -len(self) <= idx < len(self), "Index out of bounds"
        if idx < 0:
            idx = idx + len(self)
        dataset_index, dataset_internal_index = self._resolve_index(idx)
        window_data = self._sliding_window_datasets[dataset_index][dataset_internal_index]

        features = np.zeros(shape=(len(FEATURE_SIGNAL_NAMES), len(window_data.signals)), dtype=np.float)
        for i, signal_name in enumerate(FEATURE_SIGNAL_NAMES):
            raw_signal_data = window_data.signals[signal_name].values
            data = raw_signal_data if signal_name not in NORMALIZE_SIGNAL_NAMES else normalize_robust(raw_signal_data, center=False, scale=True)
            features[i, :] = data

        gt = np.array([gt.value for gt in window_data.ground_truth], dtype=np.int)

        # Convert samples into tensors
        features_tensor = torch.from_numpy(features).type(torch.float32)
        gt_tensor = torch.tensor(gt, dtype=torch.long)

        # Add noise
        if self.config.noise_mean_std is not None:
            noise = torch.normal(mean=self.config.noise_mean_std[0], std=self.config.noise_mean_std[1],
                                 size=features_tensor.size())
            features_tensor += noise

        # A few sanity checks before releasing the data to the outside world
        assert not torch.any(torch.isnan(features_tensor)) and not torch.any(torch.isnan(features_tensor)), \
            f"Oops, there's something NaN! idx={idx}, dataset_index={dataset_index}, dataset_internal_index=" \
            f"{dataset_internal_index}, dataset_name='{self._sliding_window_datasets[dataset_index].dataset_name}'"
        assert not torch.any(torch.isinf(features_tensor)) and not torch.any(torch.isinf(features_tensor)), \
            f"Oops, there's something inf! idx={idx}, dataset_index={dataset_index}, dataset_internal_index=" \
            f"{dataset_internal_index}, dataset_name='{self._sliding_window_datasets[dataset_index].dataset_name}'"

        return features_tensor, gt_tensor, idx

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

    features, gt, sample_idx = ai_dataset[0]
    pass


def test_performance(ai_dataset_provider):
    from datetime import datetime

    n_cycles = 1

    ai_dataset = ai_dataset_provider
    started_at = datetime.now()
    for c in range(n_cycles):
        for i in range(len(ai_dataset)):
            features, gt, sample_idx = ai_dataset[i]
    duration_seconds = (datetime.now()-started_at).total_seconds()

    print()
    print()
    print(f"Overall execution time: {duration_seconds:.1f}s")
    print()
    print(f"Total duration per whole dataset cycle: {duration_seconds/n_cycles:.1f}s")
    print(f"Duration per index read: {duration_seconds/n_cycles/len(ai_dataset)*1000:.2f}ms")
