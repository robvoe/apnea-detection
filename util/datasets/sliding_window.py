import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, NamedTuple, Union, Iterable
from pathlib import Path
import pickle
from enum import Enum
import functools

import pandas as pd
import numpy as np

from .physionet import read_physionet_dataset, RespiratoryEventType, RespiratoryEvent, SleepStageType


class GroundTruthClass(Enum):
    NoEvent = 0
    CentralApnea = 1
    ObstructiveApnea = 2
    MixedApnea = 3
    Hypopnea = 4


# Translation table for   RespiratoryEventType -> GroundTruthClass
_RESPIRATORY_EVENT_TYPE__GROUND_TRUTH_CLASS = {RespiratoryEventType.CentralApnea: GroundTruthClass.CentralApnea,
                                               RespiratoryEventType.MixedApnea: GroundTruthClass.MixedApnea,
                                               RespiratoryEventType.ObstructiveApnea: GroundTruthClass.ObstructiveApnea,
                                               RespiratoryEventType.Hypopnea: GroundTruthClass.Hypopnea}


WindowData = NamedTuple("WindowData", signals=pd.DataFrame, center_point=pd.Timedelta, ground_truth=Optional[pd.Series])


class SlidingWindowDataset:
    """
    Wrapper for PhysioNetDataset class. It adds the following features:
    - Preprocessing and generation of supporting data vectors
    - Caching to dramatically speed-up loading
    - Piecewise (window-based) retrieval of dataset data. Reference hereby is the center point of to-be retrieved window
    - Generation of ground truth vector. When retrieving a window, the ground truth class is delivered alongside,
      referring to the center point.
    """
    @dataclass
    class Config:
        dataset_folder: Path
        downsample_frequency_hz: float
        time_window_size: pd.Timedelta  # Length of the slided window. The outputted GT refers to its central point.
        time_window_stride: Union[pd.Timedelta, int] = 1  # Step width that we proceed with when outputting time window & ground truth vector
        ground_truth_vector_width: Union[pd.Timedelta, int] = 1  # Width of the outputted GT vector.  If 'int' is passed: Must be a positive odd number!

        def __get_index_steps(self, value: pd.Timedelta, variable_name: str) -> int:
            reference_timedelta_ = pd.to_timedelta(f"{1/self.downsample_frequency_hz * 1_000_000}us")
            index_steps_: float = value/reference_timedelta_
            assert int(index_steps_) == index_steps_, \
                f"Parameter '{variable_name}' ({value}) has no common factor with the given down-sample frequency ({self.downsample_frequency_hz} Hz)!"
            return int(index_steps_)

        def __post_init__(self):
            self.preprocessed_dataset_file = self.dataset_folder.resolve() / "preprocessed.pkl"

            # Determine all the regarding index steps out of the given parameters
            window_index_steps_ = self.__get_index_steps(value=self.time_window_size, variable_name="time_window_size")
            if (window_index_steps_ % 2) == 0:
                window_index_steps_ += 1
            self.time_window_size__index_steps = window_index_steps_

            if isinstance(self.time_window_stride, pd.Timedelta):
                self.time_window_stride__index_steps = self.__get_index_steps(value=self.time_window_stride, variable_name="time_window_stride")
            elif isinstance(self.time_window_stride, int):
                self.time_window_stride__index_steps = self.time_window_stride
            else:
                raise NotImplementedError

            if isinstance(self.ground_truth_vector_width, pd.Timedelta):
                gt_index_steps_ = self.__get_index_steps(value=self.ground_truth_vector_width, variable_name="ground_truth_vector_width")
                if (gt_index_steps_ % 2) == 0:
                    gt_index_steps_ += 1  # For ground_truth_vector_width, we always wish to work with odd numbers!
            elif isinstance(self.ground_truth_vector_width, int):
                gt_index_steps_ = self.ground_truth_vector_width
            else:
                raise NotImplementedError
            assert gt_index_steps_ > 0 and (gt_index_steps_ % 2) == 1, \
                "When passing 'ground_truth_vector_width' as int, it must be a positive odd integer!"
            self.ground_truth_vector_width__index_steps = gt_index_steps_

    def __init__(self, config: Config, allow_caching: bool = True):
        self.config = config
        assert config.dataset_folder.exists() and config.dataset_folder.is_dir(), \
            f"Given dataset folder '{config.dataset_folder.resolve()}' either not exists or is no folder."

        # Let's see if there is a cached version that we can load
        if allow_caching:
            success = self._try_read_preprocessed_dataset()
            if success:
                return

        # Load the PhysioNet dataset from disk and apply some pre-processing
        ds = read_physionet_dataset(dataset_folder=config.dataset_folder)
        ds = ds.pre_clean().downsample(target_frequency=config.downsample_frequency_hz)
        self.signals = ds.signals[["ABD", "CHEST", "AIRFLOW", "SaO2"]]
        self.respiratory_events = ds.respiratory_events
        self.sleep_stage_events = ds.sleep_stage_events
        del ds

        # Some examinations
        assert self.signals.index[0] <= config.time_window_size < self.signals.index[-1], \
            f"Chosen time_window_size '{config.time_window_size}' is too large for the given PhysioNet dataset!"

        # Determine some meta data
        dist_ = int(max(self.config.time_window_size__index_steps/2, self.config.ground_truth_vector_width__index_steps/2))
        dist_ = max(2, dist_)  # Must be at least 2 to produce reasonable values
        self._valid_center_points: pd.TimedeltaIndex = self.signals.index[dist_:-dist_:self.config.time_window_stride__index_steps]
        self._idx__signal_int_index: List[int] = list(range(len(self.signals))[dist_:-dist_:self.config.time_window_stride__index_steps])
        assert len(self._valid_center_points) == len(self._idx__signal_int_index)

        # In case there are respiratory event annotations, generate our GroundTruth vector
        self.ground_truth_vector: Optional[pd.Series] = None
        if self.respiratory_events is not None:
            gt_vector = self._generate_ground_truth_vector(signals_time_index=self.signals.index, respiratory_events=self.respiratory_events)
            assert len(gt_vector) == len(self.signals)
            # Erase beginning/ending of our gt vector, length depending on our time-window-size & gt-vector-width
            edge_cut_indexes_lr = dist_ - int(self.config.ground_truth_vector_width__index_steps/2) - 1
            gt_vector[:edge_cut_indexes_lr + 1] = np.nan
            gt_vector[-edge_cut_indexes_lr:] = np.nan
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
    def _generate_ground_truth_vector(signals_time_index: pd.TimedeltaIndex, respiratory_events: List[RespiratoryEvent]) -> pd.Series:
        gt_vector = np.ndarray(shape=(len(signals_time_index),))
        gt_vector[:] = GroundTruthClass.NoEvent.value
        for event in respiratory_events:
            start_idx = signals_time_index.get_loc(key=event.start, method="nearest")
            end_idx = signals_time_index.get_loc(key=event.end, method="nearest")
            assert event.event_type in _RESPIRATORY_EVENT_TYPE__GROUND_TRUTH_CLASS.keys(), \
                f"{event.event_type.name} seems not present in above dictionary (and likely in GroundTruthClass)"
            gt_class = _RESPIRATORY_EVENT_TYPE__GROUND_TRUTH_CLASS[event.event_type]
            gt_vector[start_idx:end_idx] = gt_class.value

        gt_series = pd.Series(data=gt_vector, index=signals_time_index, dtype="uint8")
        return gt_series

    def has_ground_truth(self):
        return self.respiratory_events is not None

    def _try_read_preprocessed_dataset(self) -> bool:
        if not self.config.preprocessed_dataset_file.is_file() or not self.config.preprocessed_dataset_file.exists():
            return False
        try:
            with open(file=self.config.preprocessed_dataset_file, mode="rb") as file:
                preprocessed_dataset: SlidingWindowDataset = pickle.load(file)
            # Check if configs match (hereby leave out the dataset folder!)
            config_1_, config_2_ = copy.deepcopy(preprocessed_dataset.config), copy.deepcopy(self.config)
            config_1_.dataset_folder = config_1_.preprocessed_dataset_file = None
            config_2_.dataset_folder = config_2_.preprocessed_dataset_file = None
            assert config_1_ == config_2_
            # Now, retrieve the cached data fields
            self._valid_center_points = preprocessed_dataset._valid_center_points
            self._idx__signal_int_index = preprocessed_dataset._idx__signal_int_index
            self.ground_truth_vector = preprocessed_dataset.ground_truth_vector
            self.respiratory_events = preprocessed_dataset.respiratory_events
            self.sleep_stage_events = preprocessed_dataset.sleep_stage_events
            self.signals = preprocessed_dataset.signals
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # We intentionally catch everything here!
            return False
        return True

    def __getitem__(self, idx: int) -> WindowData:
        assert -len(self) <= idx < len(self), "Index out of bounds"
        center_point_index = self._idx__signal_int_index[idx]
        center_point_timedelta = self._valid_center_points[idx]

        features = self.signals.iloc[center_point_index-int(self.config.time_window_size__index_steps/2):center_point_index+int(self.config.time_window_size__index_steps/2)+1]
        assert len(features) == self.config.time_window_size__index_steps

        gt_series = None
        if self.has_ground_truth():
            gt_numbers = self.ground_truth_vector[center_point_index-int(self.config.ground_truth_vector_width__index_steps/2):center_point_index+int(self.config.ground_truth_vector_width__index_steps/2)+1]
            assert len(gt_numbers) == self.config.ground_truth_vector_width__index_steps
            assert not np.any(np.isnan(gt_numbers))
            gt_classes = [GroundTruthClass(int(g)) for g in gt_numbers]
            gt_series = pd.Series(data=gt_classes, index=gt_numbers.index, name="Ground truth")
            # center_point__gt_class = self.ground_truth_vector.iloc[center_point__index]
            # assert not np.isnan(center_point__gt_class)
            # center_point__gt_class = GroundTruthClass(int(center_point__gt_class))
        return WindowData(signals=features, center_point=center_point_timedelta, ground_truth=gt_series)

    def __len__(self):
        return len(self._valid_center_points)

    @functools.cached_property
    def valid_center_points(self) -> pd.TimedeltaIndex:
        """
        Provides the range of valid center points. Center point refers to the middle of the configured time window.
        """
        return copy.deepcopy(self._valid_center_points)

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
            valid_start_ = self._valid_center_points[0]
            valid_end_ = self._valid_center_points[-1]
            assert valid_start_ <= center_point <= valid_end_, \
                f"Given center point {center_point} not in range of valid center points ({valid_start_}..{valid_end_})!"
            idx = self._valid_center_points.get_loc(center_point, method="nearest")
            assert 0 <= idx < len(self)
        else:
            idx = raw_index
        return self[idx]


def test_sliding_window_dataset():
    from util.paths import DATA_PATH

    config = SlidingWindowDataset.Config(
        dataset_folder=DATA_PATH / "training" / "tr03-0005",
        downsample_frequency_hz=5,
        time_window_size=pd.Timedelta("5 minutes"),
        time_window_stride=11,
        ground_truth_vector_width=11
    )
    sliding_window_dataset = SlidingWindowDataset(config=config, allow_caching=False)
    len_ = len(sliding_window_dataset)
    window_data = sliding_window_dataset[-1]

    valid_center_points = sliding_window_dataset.valid_center_points
    window_data_ = sliding_window_dataset.get(center_point=valid_center_points[0])
    window_data_ = sliding_window_dataset.get(center_point=pd.Timedelta("0 days 00:15:15.930000"))
    pass
