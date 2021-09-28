from typing import List, Tuple, Optional
from enum import Enum
import os
import multiprocessing as mp
import functools

import pandas as pd
import numpy as np
import numba

from util.mathutil import PeakType, IntRange, get_peaks, Peak
from util.datasets import RespiratoryEvent, RespiratoryEventType


_NECESSARY_COLUMNS = ("AIRFLOW", "ABD", "CHEST", "SaO2")


class _CoarseRespiratoryEventType(Enum):
    Apnea = 0
    Hypopnea = 1


@numba.jit(nopython=True)
def _detect_airflow_resp_events(airflow_vector: np.ndarray, sample_frequency_hz: float) -> Tuple[List[IntRange], List[_CoarseRespiratoryEventType]]:
    """Takes a look at the AIRFLOW signal and determines areas of (hypo) apneas."""
    min_event_length = 10*sample_frequency_hz
    max_event_length = 100*sample_frequency_hz
    moving_baseline_window_lr = 200  # specifies each direction (left/right) from current peak_index position
    filter_kernel_width = int(sample_frequency_hz*0.7)
    n_reference_peaks = 3
    peaks: List[Peak] = get_peaks(waveform=airflow_vector, filter_kernel_width=filter_kernel_width)

    event_areas: List[IntRange] = []
    coarse_event_types: List[_CoarseRespiratoryEventType] = []
    peak_index = 0
    while peak_index < len(peaks)-n_reference_peaks:
        # Determine a moving baseline for values that surround our current peak_index-position
        moving_window_left_index = max(peak_index - moving_baseline_window_lr, 0)
        moving_window_right_index = min(peak_index + moving_baseline_window_lr, len(peaks))
        window_peaks = peaks[moving_window_left_index:moving_window_right_index]
        moving_baseline = np.median(np.array([abs(p.extreme_value) for p in window_peaks]))
        # moving_baseline = np.sqrt(np.mean(np.array([np.square(p.extreme_value) for p in window_peaks])))
        max_allowed_moving_baseline_value = 1.0 * moving_baseline

        # Determine the reference-peaks-baseline defined by the peaks directly at our current peak_index-position
        reference_peaks = peaks[peak_index:peak_index + n_reference_peaks]
        reference_peaks_baseline = np.sqrt(np.mean(np.array([np.square(abs(p.extreme_value)) for p in reference_peaks])))
        # Determine how many subsequent peaks we need to cover at least 10s
        head_index = peak_index + n_reference_peaks
        lengths_right = np.array([p.length for p in peaks[head_index:]]).cumsum()
        cumulated_lengths_larger_than = (lengths_right >= min_event_length)
        if np.sum(cumulated_lengths_larger_than) == 0:
            break
        tail_index = head_index + cumulated_lengths_larger_than.argmax()
        # Try to stretch the window longer, whilst preserving its baseline smaller than our above determined baselines
        window_peaks = peaks[head_index:tail_index + 1]
        outside_moving_baseline = np.array([abs(p.extreme_value) for p in window_peaks]) > max_allowed_moving_baseline_value
        ratio_outside_moving_baseline = np.sum(outside_moving_baseline) / outside_moving_baseline.shape[0]
        window_baseline = np.max(np.array([abs(p.extreme_value) for p in window_peaks]))
        if window_baseline > reference_peaks_baseline * 0.7 or ratio_outside_moving_baseline > 0.5:
            peak_index += 1
            continue
        for tail_index in range(tail_index+1, len(peaks)):
            window_peaks = peaks[head_index:tail_index + 1]
            outside_moving_baseline = np.array([abs(p.extreme_value) for p in window_peaks]) > max_allowed_moving_baseline_value
            ratio_outside_moving_baseline = np.sum(outside_moving_baseline) / outside_moving_baseline.shape[0]
            window_baseline = np.max(np.array([abs(p.extreme_value) for p in window_peaks]))
            if window_baseline > reference_peaks_baseline * 0.7 or ratio_outside_moving_baseline > 0.5:
                tail_index -= 1
                break
        # Window longer than allowed? Smells like a false-positive!
        window_length = peaks[tail_index].end - peaks[head_index].start + 1
        if window_length > max_event_length:
            peak_index += 1
            continue
        # Window tail is stretched. Now make sure the signal rises up to its initial (high) value afterwards again
        post_tail_max_peak_index = min(len(peaks), tail_index+10)
        post_tail_peaks = peaks[tail_index:post_tail_max_peak_index]
        if np.sum(np.array([abs(p.extreme_value) > reference_peaks_baseline*0.9 for p in post_tail_peaks])) == 0:
            peak_index += 1
            continue
        # Now, let the beginning of the window reach to the most-negative dip right before
        pre_head_peaks = peaks[head_index-1:head_index+2]
        min_index = np.argmin(np.array([p.extreme_value for p in pre_head_peaks]))
        most_negative_dip_index = head_index - 1 + min_index
        # Determine the coarse type of our respiratory event:  Apnea/Hypopnea
        window_peaks = peaks[head_index:tail_index + 1]
        window_baseline = np.percentile(np.array([abs(p.extreme_value) for p in window_peaks]), 15)
        type_decision_reference_peaks_baseline = np.max(np.array([(abs(p.extreme_value)) for p in reference_peaks]))
        coarse_event_type: _CoarseRespiratoryEventType = _CoarseRespiratoryEventType.Apnea \
            if window_baseline <= 0.1 * type_decision_reference_peaks_baseline else _CoarseRespiratoryEventType.Hypopnea
        coarse_event_types.append(coarse_event_type)

        start = peaks[most_negative_dip_index].center
        end = peaks[tail_index].end
        event_areas.append(IntRange(start=start, end=end, length=end - start + 1))

        peak_index = tail_index
    return event_areas, coarse_event_types


def _get_peak_index(time: int, peaks: List[Peak]) -> Optional[int]:
    """Determines a peak's index within its list, depending on the given time"""
    peak_endings = np.array([p.end for p in peaks])
    flipped_index = (peak_endings < time)[::-1].argmax()
    if flipped_index == 0:
        return None
    index = len(peaks) - flipped_index
    if peaks[index].start <= time:
        return index
    return None


def _get_pre_event_peaks(event_start_time: int, peaks: List[Peak], n_peaks: int) -> List[Peak]:
    """
    Returns peaks that occurred directly before a given event.

    @param event_start_time: Start index (with respect to signal) of the event.
    @param peaks: Peaks list of the entire signal. Pre-event peaks will be taken directly from here.
    @param n_peaks: Number of pre-event peaks that we wish to get.
    @return: List of pre-event peaks.
    """
    if event_start_time > peaks[-1].end:
        return peaks

    event_start_peak_index = _get_peak_index(time=event_start_time, peaks=peaks)
    if event_start_peak_index is None:
        return []

    pre_event__end_index = event_start_peak_index - 1
    if pre_event__end_index < 0:
        return []
    pre_event__start_index = max(0, pre_event__end_index-n_peaks+1)
    pre_event_peaks = [p for p in peaks[pre_event__start_index:pre_event__end_index + 1]]
    return pre_event_peaks


def _classify_apnea(apnea_time_range: IntRange, abd_peaks: List[Peak], chest_peaks: List[Peak]) -> RespiratoryEventType:
    """
    Classifies an already-detected apnea (no hypopnea!) upon ABD and CHEST signals

    @param apnea_time_range: Time range (with respect to our AIRFLOW/ABD/CHEST/etc. signals) of the apnea.
    @param abd_peaks: Peaks list of the entire ABD signal.
    @param chest_peaks: Peaks list of the entire CHEST signal.
    @return: Classified type of the respiratory event.
    """
    n_pre_apnea_peaks = 10
    baseline_range_factor__part_1 = 3/5
    baseline_range_factor__part_2 = 2/5
    baseline_threshold_factor = 0.25
    density_threshold_factor = 0.6
    # Determine our pre-event baseline for both ABD and CHEST signals
    pre_apnea_abd_peaks = _get_pre_event_peaks(event_start_time=apnea_time_range.start, peaks=abd_peaks, n_peaks=n_pre_apnea_peaks)
    pre_apnea_chest_peaks = _get_pre_event_peaks(event_start_time=apnea_time_range.start, peaks=chest_peaks, n_peaks=n_pre_apnea_peaks)
    pre_apnea_abd_baseline = np.median(np.array([abs(p.extreme_value) for p in pre_apnea_abd_peaks]))
    pre_apnea_chest_baseline = np.median(np.array([abs(p.extreme_value) for p in pre_apnea_chest_peaks]))
    # Determine ABD and CHEST mid-apnea peaks
    apnea_abd_peaks = abd_peaks[_get_peak_index(time=apnea_time_range.start, peaks=abd_peaks):_get_peak_index(time=apnea_time_range.end, peaks=abd_peaks)]
    apnea_chest_peaks = chest_peaks[_get_peak_index(time=apnea_time_range.start, peaks=chest_peaks):_get_peak_index(time=apnea_time_range.end, peaks=chest_peaks)]
    if len(apnea_abd_peaks) < 4 or len(apnea_chest_peaks) < 4:
        return RespiratoryEventType.CentralApnea  # Short-cut to prevent division-by-0 errors in the next lines
    # Determine the mid-event baselines for both of the signals
    apnea_abd_baseline__part_1 = np.median(np.array([abs(p.extreme_value) for p in apnea_abd_peaks[:int(len(apnea_abd_peaks)*baseline_range_factor__part_1)]]))
    apnea_abd_baseline__part_2 = np.median(np.array([abs(p.extreme_value) for p in apnea_abd_peaks[int(len(apnea_abd_peaks)*baseline_range_factor__part_2):]]))
    apnea_chest_baseline__part_1 = np.median(np.array([abs(p.extreme_value) for p in apnea_chest_peaks[:int(len(apnea_chest_peaks)*baseline_range_factor__part_1)]]))
    apnea_chest_baseline__part_2 = np.median(np.array([abs(p.extreme_value) for p in apnea_chest_peaks[int(len(apnea_chest_peaks)*baseline_range_factor__part_2):]]))
    # Determine the density of peaks (per time quantum)
    pre_apnea_abd__peak_density = len(pre_apnea_abd_peaks) / (pre_apnea_abd_peaks[-1].end - pre_apnea_abd_peaks[0].start)
    pre_apnea_chest__peak_density = len(pre_apnea_chest_peaks) / (pre_apnea_chest_peaks[-1].end - pre_apnea_chest_peaks[0].start)
    apnea_abd__peak_density = len(apnea_abd_peaks) / (apnea_abd_peaks[-1].end - apnea_abd_peaks[0].start)
    apnea_chest__peak_density = len(apnea_chest_peaks) / (apnea_chest_peaks[-1].end - apnea_chest_peaks[0].start)

    # Now let's specify the ApneaType
    def _is_mixed(pre_event_baseline, event_part_1, event_part_2) -> bool:
        return event_part_1 < pre_event_baseline * baseline_threshold_factor and event_part_2 >= event_part_1 * 2.5  #  pre_event_baseline * threshold_factor  # noqa

    if _is_mixed(pre_apnea_abd_baseline, apnea_abd_baseline__part_1, apnea_abd_baseline__part_2) and \
            _is_mixed(pre_apnea_chest_baseline, apnea_chest_baseline__part_1, apnea_chest_baseline__part_2):
        return RespiratoryEventType.MixedApnea
    if all(b < pre_apnea_abd_baseline * baseline_threshold_factor for b in (apnea_abd_baseline__part_1, apnea_abd_baseline__part_2)) and \
            all(b < pre_apnea_chest_baseline * baseline_threshold_factor for b in (apnea_chest_baseline__part_1, apnea_chest_baseline__part_2)):
        return RespiratoryEventType.CentralApnea
    if apnea_abd__peak_density < pre_apnea_abd__peak_density * density_threshold_factor and apnea_chest__peak_density < pre_apnea_chest__peak_density * density_threshold_factor:
        return RespiratoryEventType.CentralApnea
    return RespiratoryEventType.ObstructiveApnea


def detect_respiratory_events(signals: pd.DataFrame, sample_frequency_hz: float, awake_series: pd.Series = None) -> List[RespiratoryEvent]:
    """
    Detects respiratory events within a bunch of given signals.

    @param signals: Signals dataframe, necessary columns are "AIRFLOW", "ABD", "CHEST", "SaO2".
    @param sample_frequency_hz: Sample frequency of given signals.
    @param awake_series: If a valid series is passed here, all detected respiratory events during wake stages (value==1)
                         will be discarded. If None is passed, no wake stages will be taken into account.
    @return: List of detected respiratory events.
    """
    assert all([col in signals for col in _NECESSARY_COLUMNS]), \
        f"At least one of the necessary columns ({_NECESSARY_COLUMNS}) is missing in the passed DataFrame"
    if awake_series is not None:
        assert awake_series.index == signals.index, "Indexes of both 'signals' and 'is_awake' must be equal!"
    ranges, coarse_respiratory_event_types = _detect_airflow_resp_events(airflow_vector=signals["AIRFLOW"].values, sample_frequency_hz=sample_frequency_hz)
    chest_peaks = get_peaks(waveform=signals["CHEST"].values, filter_kernel_width=int(sample_frequency_hz*0.7))
    abd_peaks = get_peaks(waveform=signals["ABD"].values, filter_kernel_width=int(sample_frequency_hz * 0.7))

    apnea_events: List[RespiratoryEvent] = []
    n_discarded_wake_stages = 0
    n_filtered_hypopneas = 0
    for range, coarse_type in zip(ranges, coarse_respiratory_event_types):
        if awake_series is not None:
            is_awake_values = awake_series.values != 0
            range_mat = np.zeros(shape=(len(signals),), dtype=bool)
            range_mat[range.start:range.end] = True
            if np.sum(is_awake_values & range_mat) != 0:
                n_discarded_wake_stages += 1
                continue
        start = signals.index[range.start]
        end = signals.index[range.end]

        # If Hypopnea was detected, make sure SaO2 value falls accordingly by >= 3%
        if coarse_type == _CoarseRespiratoryEventType.Hypopnea:
            max_pre_event_sa_o2 = signals["SaO2"][start-pd.to_timedelta("30s"):start].max()
            min_post_event_sa_o2 = signals["SaO2"][start:end+pd.to_timedelta("60s")].min()
            if not min_post_event_sa_o2 <= max_pre_event_sa_o2*0.97:
                n_filtered_hypopneas += 1
                continue

        # If an apnea was detected, now further specify its type
        event_type = RespiratoryEventType.Hypopnea
        if coarse_type == _CoarseRespiratoryEventType.Apnea:
            event_type = _classify_apnea(apnea_time_range=range, abd_peaks=abd_peaks, chest_peaks=chest_peaks)
            assert event_type != RespiratoryEventType.Hypopnea, "Function _classify_apnea() must not return HypoApnea type!"

        apnea_events += [RespiratoryEvent(start=start, end=end, aux_note=None, event_type=event_type)]

    if awake_series:
        print(f"Discarded {n_discarded_wake_stages} detected respiratory events, as they overlap with wake stages")
    print(f"Discarded {n_filtered_hypopneas} hypopneas, due to SaO2 not falling by 3%")
    return apnea_events


def _mp_exec_fn(signal_awake: Tuple[pd.DataFrame, Optional[pd.Series]], sample_frequency_hz: float):
    """Just an internal helper function. Wraps multicore access."""
    return detect_respiratory_events(signals=signal_awake[0], sample_frequency_hz=sample_frequency_hz, awake_series=signal_awake[1])


def detect_respiratory_events_multicore(signals: List[pd.DataFrame], sample_frequency_hz: float,
                                        awake_series: List[Optional[pd.Series]] = None, progress_fn=None,
                                        n_processes: int = None) -> List[List[RespiratoryEvent]]:
    """
    Essentially the same as the function detect_respiratory_events, just that its heavy calculations will be performed
    on multiple CPU cores.

    @param signals: List of signals dataframes, each standing for one dataset. Necessary columns are "AIRFLOW", "ABD",
                    "CHEST", "SaO2".
    @param sample_frequency_hz: Sample frequency of given signals.
    @param awake_series: If a valid series is passed here, all detected respiratory events during wake stages (value==1)
                         will be discarded. If None is passed, no wake stages will be taken into account. If a list is
                         passed, its length must match the length of signals list. Single list elements may be None.
    @param progress_fn: Function that may print prediction progress, e.g. tqdm. If None, no progress will be shown.
    @param n_processes: Number of processes we wish spread the work to. If None, an optimum will be chosen.

    @return: A list of the same length as the signals list.
    """
    # At first, make sure everything's in place
    assert len(signals) > 0, "Empty signals list was passed"

    # Find a few default values
    if awake_series is None:
        awake_series = [None] * len(signals)
    assert len(signals) == len(awake_series), \
        f"Length of passed 'awake_series' list ({len(awake_series)}) differs from 'signals' list ({len(signals)})"

    affinity = len(os.sched_getaffinity(0))
    if n_processes is None:
        n_processes = max(1, affinity - 2)
    assert 1 <= n_processes <= affinity, f"Given 'n_processes' not in the allowed range of 1..{affinity}"

    if progress_fn is None:
        def progress_fn(x): return x

    # Let's get started
    with mp.Pool(processes=n_processes) as pool:
        load_fn_ = functools.partial(_mp_exec_fn, sample_frequency_hz=sample_frequency_hz)
        loading_results = list(progress_fn(pool.imap(load_fn_, zip(signals, awake_series))))
        results: List[List[RespiratoryEvent]] = loading_results
    return results


def test_development():
    from util.datasets import SlidingWindowDataset
    from util.paths import DATA_PATH

    config = SlidingWindowDataset.Config(
        downsample_frequency_hz=5,
        time_window_size=pd.Timedelta("2 minutes")
    )
    sliding_window_dataset = SlidingWindowDataset(config=config, dataset_folder=DATA_PATH / "training" / "tr03-0005", allow_caching=True)

    events = detect_respiratory_events(signals=sliding_window_dataset.signals, sample_frequency_hz=config.downsample_frequency_hz, awake_series=None)
    pass


def test_development__multiprocessing():
    from util.datasets import SlidingWindowDataset
    from util.paths import DATA_PATH

    config = SlidingWindowDataset.Config(
        downsample_frequency_hz=5,
        time_window_size=pd.Timedelta("2 minutes"),
    )
    sliding_window_dataset = SlidingWindowDataset(config=config, dataset_folder=DATA_PATH / "training" / "tr03-0005", allow_caching=True)

    events_lists = detect_respiratory_events_multicore(signals=[sliding_window_dataset.signals], sample_frequency_hz=config.downsample_frequency_hz, awake_series=None)
    pass
