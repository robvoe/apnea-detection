from typing import List, Optional, Dict
from pathlib import Path

import pandas as pd
import wfdb

from .definitions import RespiratoryEventType, RespiratoryEvent, EnduringEvent, PhysioNetDataset, _Event, TransientEvent, \
    SleepStageType, SleepStageEvent


__author__ = "Robert Voelckner"
__copyright__ = "Copyright 2021"
__license__ = "MIT"


_SLEEP_STAGE_KEYWORDS = [s.value for s in SleepStageType]


def _create_enduring_event(start: pd.Timedelta, end: pd.Timedelta, aux_note: str) -> EnduringEvent:
    """
    Creates an EnduringEvent out of the given parameters. In certain cases, it chooses the sub-class ApneaEvent instead.
    """
    if "pnea" in aux_note:
        if "centralapnea" in aux_note:
            event_type = RespiratoryEventType.CentralApnea
        elif "mixedapnea" in aux_note:
            event_type = RespiratoryEventType.MixedApnea
        elif "obstructiveapnea" in aux_note:
            event_type = RespiratoryEventType.ObstructiveApnea
        elif "hypopnea" in aux_note:
            event_type = RespiratoryEventType.Hypopnea
        else:
            raise RuntimeError(f"Unrecognized *pnea event aux_note: '{aux_note}'")
        return RespiratoryEvent(start=start, end=end, aux_note=aux_note, event_type=event_type)
    return EnduringEvent(start=start, end=end, aux_note=aux_note)


def _create_transient_event(start: pd.Timedelta, aux_note: str) -> TransientEvent:
    if aux_note in _SLEEP_STAGE_KEYWORDS:
        sleep_stage_type = SleepStageType(aux_note)
        return SleepStageEvent(start=start, aux_note=aux_note, sleep_stage_type=sleep_stage_type)
    return TransientEvent(start=start, aux_note=aux_note)


def read_physionet_dataset(dataset_folder: Path, dataset_filename_stem: str = None) -> PhysioNetDataset:
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
    df_signals = pd.DataFrame(data=record.p_signal, columns=record.sig_name, index=index, dtype="float32")
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
                events += [_create_enduring_event(start=index[start_sample_idx], end=index[sample_idx], aux_note=aux_note)]
            else:
                events += [_create_transient_event(start=index[sample_idx], aux_note=aux_note)]
    return PhysioNetDataset(signals=df_signals, signal_units=signal_units, sample_frequency_hz=sample_frequency,
                            events=events)


def test_read_dataset():
    from util.paths import DATA_PATH
    dataset = read_physionet_dataset(dataset_folder=DATA_PATH / "training" / "tr03-0005")
    respiratory_events = dataset.respiratory_events
    dataset = dataset.pre_clean()
    dataset = dataset.downsample(downsampling_factor=10)
    pass
