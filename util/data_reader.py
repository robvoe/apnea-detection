from dataclasses import dataclass
from typing import Union, List, Optional, Set, Dict
from pathlib import Path
from abc import ABC

import pandas as pd
import wfdb


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
    signal: pd.DataFrame
    signal_units: List[str]
    sample_frequency_hz: float
    events: Optional[List[_Event]]  # May be None in case there is no event list (i.e. arousal file)


def read_dataset(dataset_folder: Path, dataset_filename_stem: str = None) -> Dataset:
    """
    Reads samples and annotations from a dataset folder. Annotations are optional, so only parsed when present in
    the given folder.

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
    df_signal = pd.DataFrame(data=record.p_signal, columns=record.sig_name, index=index)
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
    return Dataset(signal=df_signal, signal_units=signal_units, sample_frequency_hz=sample_frequency, events=events)


def test_read_dataset():
    from util.paths import DATA_PATH
    dataset = read_dataset(dataset_folder=DATA_PATH / "training" / "tr03-0005")
    pass
