from dataclasses import dataclass
from typing import Optional, Tuple, List, NamedTuple
from pathlib import Path
import pickle
from enum import Enum
import functools

import pandas as pd
import numpy as np

from .physionet import read_physionet_dataset, ApneaType, ApneaEvent, SleepStageType

__author__ = "Robert Voelckner"
__copyright__ = "Copyright 2021"
__license__ = "MIT"


class GroundTruthClass(Enum):
    NoApnea = 0
    CentralApnea = 1
    ObstructiveApnea = 2
    MixedApnea = 3
    HypoApnea = 4


# Translation table for   ApneaType -> GroundTruthClass
_APNEA_TYPE__GROUND_TRUTH_CLASS = {ApneaType.CentralApnea: GroundTruthClass.CentralApnea,
                                   ApneaType.MixedApnea: GroundTruthClass.MixedApnea,
                                   ApneaType.ObstructiveApnea: GroundTruthClass.ObstructiveApnea,
                                   ApneaType.HypoApnea: GroundTruthClass.HypoApnea}


WindowData = NamedTuple("WindowData", signals=pd.DataFrame, center_point=pd.Timedelta, ground_truth_class=Optional[GroundTruthClass])


class SlidingWindowDataset:
    @dataclass
    class Config:
        physionet_dataset_folder: Path
        downsample_frequency_hz: float
        time_window_size: pd.Timedelta  # Length of the slided window. The outputted GT refers to its central point.

        def __post_init__(self):
            self.preprocessed_dataset_file = self.physionet_dataset_folder.resolve() / "preprocessed.pkl"

    def __init__(self, config: Config, allow_caching: bool = True):
        self.config = config
        assert config.physionet_dataset_folder.exists() and config.physionet_dataset_folder.is_dir(), \
            f"Given dataset folder '{config.physionet_dataset_folder.resolve()}' either not exists or is no folder."

        # Let's see if there is a cached version that we can load
        if allow_caching:
            success = self._try_read_preprocessed_dataset()
            if success:
                return

        # Load the PhysioNet dataset from disk and apply some pre-processing
        ds = read_physionet_dataset(dataset_folder=config.physionet_dataset_folder)
        ds = ds.pre_clean().downsample(target_frequency=config.downsample_frequency_hz)
        self.signals = ds.signals[["ABD", "CHEST", "AIRFLOW", "SaO2"]]
        self.apnea_events = ds.apnea_events
        self.sleep_stage_events = ds.sleep_stage_events
        del ds

        # Some examinations
        assert self.signals.index[0] <= config.time_window_size < self.signals.index[-1], \
            f"Chosen time_window_size '{config.time_window_size}' is too large for the given PhysioNet dataset!"
        self._time_window_size__index_steps = self.signals.index.get_loc(config.time_window_size, method="pad")
        self._n_window_steps = len(self.signals.index) - self._time_window_size__index_steps + 1

        # In case there are apnea event annotations, generate our GroundTruth vector
        self.ground_truth_vector: Optional[pd.Series] = None
        if self.apnea_events is not None:
            gt_vector = self._generate_ground_truth_vector(temporal_index=self.signals.index, apnea_events=self.apnea_events)
            # Erase beginning/ending of our gt vector, length depending on our time window size
            edge_cut_index_steps = int(self._time_window_size__index_steps / 2)
            gt_vector[:edge_cut_index_steps] = np.nan
            gt_vector[-edge_cut_index_steps + 1:] = np.nan
            self.ground_truth_vector = gt_vector

        # In case we have event annotations, generate our "awake" vector
        if self.sleep_stage_events is not None:
            is_awake_mat = np.zeros(shape=(len(self.signals.index),), dtype="int8")
            for event in self.sleep_stage_events:
                start_idx = self.signals.index.get_loc(event.start)
                value = 1 if event.sleep_stage_type == SleepStageType.Wakefulness else 0
                is_awake_mat[start_idx:] = value
            is_awake_series = pd.Series(data=is_awake_mat, index=self.signals.index)
            self.signals["Is awake (ref. sleep stages)"] = is_awake_series

        # Serialize preprocessed dataset to disk
        if allow_caching:
            with open(file=self.config.preprocessed_dataset_file, mode="wb") as file:
                pickle.dump(obj=self, file=file)

    @staticmethod
    def _generate_ground_truth_vector(temporal_index: pd.TimedeltaIndex, apnea_events: List[ApneaEvent]) -> pd.Series:
        gt_vector = np.ndarray(shape=(len(temporal_index),))
        gt_vector[:] = GroundTruthClass.NoApnea.value
        for apnea_event in apnea_events:
            start_idx = temporal_index.get_loc(key=apnea_event.start, method="nearest")
            end_idx = temporal_index.get_loc(key=apnea_event.end, method="nearest")
            assert apnea_event.apnea_type in _APNEA_TYPE__GROUND_TRUTH_CLASS.keys(), \
                f"{apnea_event.apnea_type.name} seems not present in above dictionary (and likely in GroundTruthClass)"
            gt_class = _APNEA_TYPE__GROUND_TRUTH_CLASS[apnea_event.apnea_type]
            gt_vector[start_idx:end_idx] = gt_class.value

        gt_series = pd.Series(data=gt_vector, index=temporal_index, dtype="uint8")
        return gt_series

    def has_ground_truth(self):
        return self.apnea_events is not None

    def _try_read_preprocessed_dataset(self) -> bool:
        if not self.config.preprocessed_dataset_file.is_file() or not self.config.preprocessed_dataset_file.exists():
            return False
        try:
            with open(file=self.config.preprocessed_dataset_file, mode="rb") as file:
                preprocessed_dataset: SlidingWindowDataset = pickle.load(file)
            assert preprocessed_dataset.config.time_window_size == self.config.time_window_size
            assert preprocessed_dataset.config.downsample_frequency_hz == self.config.downsample_frequency_hz
            self._time_window_size__index_steps = preprocessed_dataset._time_window_size__index_steps
            self._n_window_steps = preprocessed_dataset._n_window_steps
            self.ground_truth_vector = preprocessed_dataset.ground_truth_vector
            self.apnea_events = preprocessed_dataset.apnea_events
            self.signals = preprocessed_dataset.signals
        except BaseException:
            return False
        return True

    def __getitem__(self, idx) -> WindowData:
        assert -len(self) <= idx < len(self), "Index out of bounds"
        if idx < 0:
            idx = idx + len(self)
        features = self.signals.iloc[idx:idx+self._time_window_size__index_steps]

        center_point__index = idx + int(self._time_window_size__index_steps/2)
        center_point__timedelta = self.signals.index[center_point__index]

        center_point__gt_class = None
        if self.ground_truth_vector is not None:
            center_point__gt_class = self.ground_truth_vector.iloc[center_point__index]
            assert not np.isnan(center_point__gt_class)
            center_point__gt_class = GroundTruthClass(int(center_point__gt_class))
        return WindowData(signals=features, center_point=center_point__timedelta, ground_truth_class=center_point__gt_class)

    def __len__(self):
        return self._n_window_steps

    @functools.cached_property
    def valid_center_points(self) -> pd.TimedeltaIndex:
        """
        Provides the range of valid center points. Center point refers to the middle of the configured time window.
        """
        central_point__index = int(self._time_window_size__index_steps / 2)
        index = self.signals.index[central_point__index:-central_point__index+1]
        assert len(index) == self._n_window_steps
        return index

    def get(self, center_point: pd.Timedelta = None, raw_index: int = None) -> WindowData:
        """
        Returns values for a specific time window. The position of the time window either refers to the raw index,
        or to the center of the time window.

        :param center_point: The function returns values centered around the given center point.
        :param raw_index: The function acts exactly like __getitem__
        """
        assert (center_point is None and raw_index is not None) or (center_point is not None and raw_index is None), \
            "Exactly one of the given arguments must be None!"
        if center_point is not None:
            valid_start_ = self.valid_center_points[0]
            valid_end_ = self.valid_center_points[-1]
            assert valid_start_ <= center_point <= valid_end_, \
                f"Given center point {center_point} not in range of valid center points ({valid_start_}..{valid_end_})!"
            center_point__index = self.signals.index.get_loc(center_point, method="nearest")
            window_start__index = center_point__index - int(self._time_window_size__index_steps/2)
            assert 0 <= window_start__index < len(self)
        else:
            window_start__index = raw_index
        return self[window_start__index]


def test_sliding_window_dataset():
    from util.paths import DATA_PATH

    config = SlidingWindowDataset.Config(
        physionet_dataset_folder=DATA_PATH / "training" / "tr03-0005",
        downsample_frequency_hz=10,
        time_window_size=pd.Timedelta("2 minutes")
    )
    sliding_window_dataset = SlidingWindowDataset(config=config, allow_caching=True)
    len_ = len(sliding_window_dataset)
    window_data = sliding_window_dataset[-1]

    valid_time_range = sliding_window_dataset.valid_center_points
    window_data_ = sliding_window_dataset.get(center_point=valid_time_range[-1])
    window_data_ = sliding_window_dataset.get(center_point=pd.Timedelta("0 days 00:15:15.930000"))
    pass
