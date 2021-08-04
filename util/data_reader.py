from copy import deepcopy
from dataclasses import dataclass
import functools
from typing import Union, List, Optional, Set, Dict
from pathlib import Path
from abc import ABC

import numpy as np
import pandas as pd
import wfdb

from .filter import apply_butterworth_lowpass_filter, apply_butterworth_bandpass_filter


__author__ = "Robert Voelckner"
__copyright__ = "Copyright 2021"
__license__ = "MIT"


@dataclass
class _SampleReference:
    sample_num: int  # Refers to the plain integer-index within the signals
    sample_timedelta: pd.Timedelta  # Refers to the time-of-occurrence; meant to be used with DataFrame

    def __str__(self):
        return f"'{self.sample_timedelta}'"

    def __repr__(self):
        return str(self)

    def __int__(self):
        return self.sample_num


@dataclass
class _Event(ABC):
    start: _SampleReference
    aux_note: str


@dataclass
class TransientEvent(_Event):
    pass


@dataclass
class EnduringEvent(_Event):
    end: _SampleReference


@dataclass
class Dataset:
    signals: pd.DataFrame
    signal_units: List[str]
    sample_frequency_hz: float
    events: Optional[List[_Event]]  # May be None in case there is no event list (i.e. arousal file)

    @functools.cached_property
    def apnea_events(self) -> List[EnduringEvent]:
        assert self.events is not None, "No events available, most likely because no arousal file was parsed."
        apnea_events = list(filter(lambda evt: "apnea" in evt.aux_note and isinstance(evt, EnduringEvent), self.events))
        return apnea_events  # noqa   <-- Code-checker complains about incorrect type of list

    def clean(self) -> "Dataset":
        """
        This function cleans the signal data and applies lowpass/bandpass filtering according to AASM v2.0.3, page 12.
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
        return o

    def downsample(self, factor: int = 10) -> "Dataset":
        """
        Returns a downsampled version of the dataset. The only touched data fields are 'signals' and
        'sample_frequency_hz'.
        """
        o = deepcopy(self)
        o.sample_frequency_hz /= factor
        o.signals = o.signals.resample(rule=f"{1/o.sample_frequency_hz*1_000_000}us").mean()
        return o


def read_dataset(dataset_folder: Path, dataset_filename_stem: str = None) -> Dataset:
    """
    Reads datasets of the [*PhysioNet Challenge 2018*](https://physionet.org/content/challenge-2018/1.0.0/). Reads
    samples and annotations from a given dataset folder.
    *Annotations are optional*, so only parsed when available in the given folder.

    :param dataset_folder: The folder containing our .mat, .hea and .arousal files
    :param dataset_filename_stem: Name that all the dataset files have in common. If None, we'll derive it from the
                                  folder name.
    :return: The Dataset instance.
    """
    assert dataset_folder.is_dir() and dataset_folder.exists(), \
        f"Given dataset folder {dataset_folder} either not exists or is no folder."
    if dataset_filename_stem is None:
        dataset_filename_stem = dataset_folder.name

    # Read the signal files (.hea & .mat)
    record = wfdb.rdrecord(record_name=str(dataset_folder / dataset_filename_stem))
    sample_frequency = float(record.fs)
    index = pd.timedelta_range(start=0, periods=len(record.p_signal), freq=f"{1/sample_frequency*1_000_000}us")
    df_signals = pd.DataFrame(data=record.p_signal, columns=record.sig_name, index=index)
    signal_units = record.units

    # In case they exist, read the annotations
    arousal_file = dataset_folder / f"{dataset_filename_stem}.arousal"
    events: Optional[List[_Event]] = None
    if arousal_file.exists():
        arousal = wfdb.rdann(record_name=str(dataset_folder / dataset_filename_stem), extension="arousal")
        events = []
        open_parentheses: Dict[str, int] = {}
        for aux_note, sample_idx in zip(arousal.aux_note, arousal.sample):
            aux_note = str(aux_note).strip()
            if aux_note.startswith("("):
                aux_note = aux_note.lstrip("(")
                assert aux_note not in open_parentheses, f"Event '{aux_note}' cannot start twice!"
                open_parentheses[aux_note] = sample_idx
            elif aux_note.endswith(")"):
                aux_note = aux_note.rstrip(")")
                assert aux_note in open_parentheses, f"Event '{aux_note}' cannot end before starting!"
                start_sample_idx = open_parentheses.pop(aux_note)
                start = _SampleReference(sample_num=start_sample_idx, sample_timedelta=index[start_sample_idx])
                end = _SampleReference(sample_num=sample_idx, sample_timedelta=index[sample_idx])
                events += [EnduringEvent(start=start, end=end, aux_note=aux_note)]
            else:
                start = _SampleReference(sample_num=sample_idx, sample_timedelta=index[sample_idx])
                events += [TransientEvent(start=start, aux_note=aux_note)]
    return Dataset(signals=df_signals, signal_units=signal_units, sample_frequency_hz=sample_frequency, events=events)


def test_read_dataset():
    from util.paths import DATA_PATH
    dataset = read_dataset(dataset_folder=DATA_PATH / "training" / "tr03-0005")
    dataset = dataset.clean()
    dataset = dataset.downsample(factor=10)
    pass
