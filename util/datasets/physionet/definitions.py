from abc import ABC
from dataclasses import dataclass
import functools
from copy import deepcopy
from enum import Enum
from typing import Optional, List

import numpy as np
import pandas as pd

from util.filter import apply_butterworth_lowpass_filter, apply_butterworth_bandpass_filter


__author__ = "Robert Voelckner"
__copyright__ = "Copyright 2021"
__license__ = "MIT"


class ApneaType(Enum):
    CentralApnea = 0
    ObstructiveApnea = 1
    MixedApnea = 2
    HypoApnea = 3


class SleepStageType(Enum):  # Terminology according to AASM, v2.0.3, page 18
    Wakefulness = "W"
    NREM1 = "N1"
    NREM2 = "N2"
    NREM3 = "N3"
    REM = "R"


@dataclass
class _Event(ABC):
    start: pd.Timedelta
    aux_note: Optional[str]


@dataclass
class TransientEvent(_Event):
    pass


@dataclass
class SleepStageEvent(TransientEvent):
    sleep_stage_type: SleepStageType


@dataclass
class EnduringEvent(_Event):
    end: pd.Timedelta

    def overlaps(self, other: "EnduringEvent") -> bool:
        """Returns if two EnduringEvent instances temporally overlap."""
        if self.start <= other.start <= self.end:
            return True
        if other.start <= self.start <= other.end:
            return True
        return False


@dataclass
class ApneaEvent(EnduringEvent):
    apnea_type: ApneaType


@dataclass
class PhysioNetDataset:
    signals: pd.DataFrame
    signal_units: List[str]
    sample_frequency_hz: float
    events: Optional[List[_Event]]  # May be None in case there is no event list (i.e. arousal file)

    @functools.cached_property
    def apnea_events(self) -> Optional[List[ApneaEvent]]:
        if self.events is None:
            return None
        apnea_events = [e for e in self.events if isinstance(e, ApneaEvent)]
        return apnea_events  # noqa   <-- Code-checker complains about allegedly incorrect type of list

    @functools.cached_property
    def sleep_stage_events(self) -> Optional[List[SleepStageEvent]]:
        if self.events is None:
            return None
        sleep_stage_events = [e for e in self.events if isinstance(e, SleepStageEvent)]
        return sleep_stage_events  # noqa   <-- Code-checker complains about allegedly incorrect type of list

    def pre_clean(self) -> "PhysioNetDataset":
        """
        This function pre-cleans the signal data and applies low-/band-pass filtering according to AASM v2.0.3, page 12.

        :return: Cleaned and filtered copy of the dataset
        """
        o = deepcopy(self)
        bandpass_wrapper = functools.partial(apply_butterworth_bandpass_filter, f_sample=self.sample_frequency_hz,
                                             filter_order=5)

        sa_o2 = o.signals["SaO2"]
        sa_o2[sa_o2 <= 20.0] = np.NAN
        sa_o2 = sa_o2.interpolate(method="linear")
        o.signals["SaO2"] = sa_o2

        o.signals["AIRFLOW"] = bandpass_wrapper(o.signals["AIRFLOW"], f_low_cutoff=0.03, f_high_cutoff=3)
        o.signals["ABD"] = bandpass_wrapper(o.signals["ABD"], f_low_cutoff=0.1, f_high_cutoff=15)
        o.signals["CHEST"] = bandpass_wrapper(o.signals["CHEST"], f_low_cutoff=0.1, f_high_cutoff=15)

        eeg_eog_signals = ("F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1", "Chin1-Chin2")
        for name in eeg_eog_signals:
            o.signals[name] = bandpass_wrapper(o.signals[name], f_low_cutoff=0.3, f_high_cutoff=35)

        o.signals["ECG"] = bandpass_wrapper(o.signals["ABD"], f_low_cutoff=0.3, f_high_cutoff=70)
        o.signals = o.signals.astype("float32")
        return o

    def downsample(self, downsampling_factor: int = None, target_frequency: float = None) -> "PhysioNetDataset":
        """
        Returns a downsampled version of the dataset. The only touched data fields are 'signals' and
        'sample_frequency_hz'. Exactly one of two two parameters must be provided!
        """
        assert (downsampling_factor is None and target_frequency is not None) or \
               (downsampling_factor is not None and target_frequency is None), \
            "Exactly one of both parameters must be None!"
        o = deepcopy(self)
        if downsampling_factor is not None:
            o.sample_frequency_hz /= downsampling_factor
        else:
            o.sample_frequency_hz = target_frequency
        o.signals = o.signals.resample(rule=f"{1/o.sample_frequency_hz*1_000_000}us").mean()
        return o


def test_enduring_event_overlaps():
    event1 = EnduringEvent(start=pd.to_timedelta("1 minute"), end=pd.to_timedelta("2 minute"), aux_note=None)
    event2 = EnduringEvent(start=pd.to_timedelta("1 minute"), end=pd.to_timedelta("2 minute"), aux_note=None)
    assert event1.overlaps(event2) is True
    assert event2.overlaps(event1) is True

    event1 = EnduringEvent(start=pd.to_timedelta("0.5 minute"), end=pd.to_timedelta("1.5 minute"), aux_note=None)
    event2 = EnduringEvent(start=pd.to_timedelta("1 minute"), end=pd.to_timedelta("2 minute"), aux_note=None)
    assert event1.overlaps(event2) is True
    assert event2.overlaps(event1) is True

    event1 = EnduringEvent(start=pd.to_timedelta("0.2 minute"), end=pd.to_timedelta("0.7 minute"), aux_note=None)
    event2 = EnduringEvent(start=pd.to_timedelta("1 minute"), end=pd.to_timedelta("2 minute"), aux_note=None)
    assert event1.overlaps(event2) is False
    assert event2.overlaps(event1) is False

    event1 = EnduringEvent(start=pd.to_timedelta("1.2 minute"), end=pd.to_timedelta("1.7 minute"), aux_note=None)
    event2 = EnduringEvent(start=pd.to_timedelta("1 minute"), end=pd.to_timedelta("2 minute"), aux_note=None)
    assert event1.overlaps(event2) is True
    assert event2.overlaps(event1) is True

    event1 = EnduringEvent(start=pd.to_timedelta("1.5 minute"), end=pd.to_timedelta("2.5 minute"), aux_note=None)
    event2 = EnduringEvent(start=pd.to_timedelta("1 minute"), end=pd.to_timedelta("2 minute"), aux_note=None)
    assert event1.overlaps(event2) is True
    assert event2.overlaps(event1) is True

    event1 = EnduringEvent(start=pd.to_timedelta("2 minute"), end=pd.to_timedelta("3 minute"), aux_note=None)
    event2 = EnduringEvent(start=pd.to_timedelta("1 minute"), end=pd.to_timedelta("2 minute"), aux_note=None)
    assert event1.overlaps(event2) is True
    assert event2.overlaps(event1) is True

    event1 = EnduringEvent(start=pd.to_timedelta("3 minute"), end=pd.to_timedelta("4 minute"), aux_note=None)
    event2 = EnduringEvent(start=pd.to_timedelta("1 minute"), end=pd.to_timedelta("2 minute"), aux_note=None)
    assert event1.overlaps(event2) is False
    assert event2.overlaps(event1) is False
