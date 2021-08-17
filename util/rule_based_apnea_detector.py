from typing import List, Tuple, Optional
from enum import Enum

import pandas as pd
import numpy as np
import numba
import numba.typed

from util.mathutil import PeakType, Cluster, get_peaks, cluster_1d, Peak
from .datasets import RespiratoryEvent, RespiratoryEventType

__author__ = "Robert Voelckner"
__copyright__ = "Copyright 2021"
__license__ = "MIT"


_NECESSARY_COLUMNS = ("AIRFLOW", "ABD", "CHEST", "SaO2")


class _CoarseRespiratoryEventType(Enum):
    Apnea = 0
    Hypopnea = 1


# @numba.jit(nopython=True)
# def _detect_potential_apnea_airflow_areas(airflow_vector: np.ndarray, sample_frequency_hz: float) -> List[Cluster]:
#     filter_kernel_width = int(sample_frequency_hz*0.7)
#     peaks = get_peaks(waveform=airflow_vector, filter_kernel_width=filter_kernel_width)
#
#     # Figure out potential low-amplitude areas
#     overall_baseline = np.median(np.array([abs(p.extreme_value) for p in peaks]))
#     low_peaks = [p for p in peaks if abs(p.extreme_value) <= overall_baseline*0.9]
#     low_peaks_mat = np.zeros(shape=airflow_vector.shape)
#     for p in low_peaks:
#         low_peaks_mat[p.start:p.end] = p.extreme_value
#     low_peak_clusters = cluster_1d(low_peaks_mat, no_klass=0, allowed_distance=int(sample_frequency_hz*0.5), min_length=int(sample_frequency_hz * 7))
#
#     # Now, do some further filtering with the low-amplitude areas we found
#     # potential_apnea_clusters: List[Cluster] = []
#     # for cluster in low_peak_clusters:
#     #     cluster_peaks = [p for p in peaks if cluster.start <= p.center <= cluster.end]
#     #     n_cluster_peaks_below_baseline = len([p for p in cluster_peaks if abs(p.extreme_value) <= overall_baseline])
#     #     if n_cluster_peaks_below_baseline / len(cluster_peaks) >= 0.8:
#     #         potential_apnea_clusters.append(cluster)
#
#     # Find out, which of the detected low-amplitude areas are apneas
#     apnea_areas: List[Cluster] = []
#     for cluster in low_peak_clusters:
#         cluster_peaks = [p for p in peaks if cluster.start <= p.center <= cluster.end]
#         pre_cluster_peaks = [p for p in peaks if p.center < cluster.start]
#         if len(pre_cluster_peaks) == 0:
#             continue
#         n_pre_cluster_peaks = int(min(len(pre_cluster_peaks), 3))
#         pre_cluster_peaks = pre_cluster_peaks[-n_pre_cluster_peaks:]
#
#         cluster_baseline = np.median(np.array([abs(p.extreme_value) for p in cluster_peaks]))
#         if any([abs(p.extreme_value*0.3) >= cluster_baseline for p in pre_cluster_peaks]):
#             apnea_areas.append(cluster)
#     return apnea_areas


@numba.jit(nopython=True)
def _detect_airflow_resp_events(airflow_vector: np.ndarray, sample_frequency_hz: float) -> Tuple[List[Cluster], List[_CoarseRespiratoryEventType]]:
    """Takes a look at the AIRFLOW signal and determines areas of (hypo) apneas."""
    min_event_length = 10*sample_frequency_hz
    max_event_length = 100*sample_frequency_hz
    moving_baseline_window_lr = 200  # specifies each direction (left/right) from current peak_index position
    filter_kernel_width = int(sample_frequency_hz*0.7)
    n_reference_peaks = 3
    peaks: List[Peak] = get_peaks(waveform=airflow_vector, filter_kernel_width=filter_kernel_width)

    event_areas: List[Cluster] = []
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
        window_baseline = np.percentile(np.array([abs(p.extreme_value) for p in window_peaks]), 10)
        type_decision_reference_peaks_baseline = np.max(np.array([(abs(p.extreme_value)) for p in reference_peaks]))
        coarse_event_type: _CoarseRespiratoryEventType = _CoarseRespiratoryEventType.Apnea \
            if window_baseline <= 0.1 * type_decision_reference_peaks_baseline else _CoarseRespiratoryEventType.Hypopnea
        coarse_event_types.append(coarse_event_type)

        start = peaks[most_negative_dip_index].center
        end = peaks[tail_index].end
        event_areas.append(Cluster(start=start, end=end, length=end-start+1))

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


def _get_pre_event_peaks(event_start: int, peaks: List[Peak], n_peaks: int) -> List[Peak]:
    if event_start > peaks[-1].end:
        return peaks

    event_start_peak_index = _get_peak_index(time=event_start, peaks=peaks)
    if event_start_peak_index is None:
        return []

    pre_event__end_index = event_start_peak_index - 1
    if pre_event__end_index < 0:
        return []
    pre_event__start_index = max(0, pre_event__end_index-n_peaks+1)
    pre_event_peaks = [p for p in peaks[pre_event__start_index:pre_event__end_index + 1]]
    return pre_event_peaks


def _classify_apnea(apnea_area: Cluster, abd_peaks: List[Peak], chest_peaks: List[Peak], sample_frequency_hz: float) -> RespiratoryEventType:
    """Classifies a detected apnea (no hypopnea!) upon ABD and CHEST signals"""
    n_pre_apnea_peaks = 10
    baseline_range_factor__part_1 = 3/5
    baseline_range_factor__part_2 = 2/5
    baseline_threshold_factor = 0.25
    density_threshold_factor = 0.6
    # Determine our pre-event baseline for both ABD and CHEST signals
    pre_apnea_abd_peaks = _get_pre_event_peaks(event_start=apnea_area.start, peaks=abd_peaks, n_peaks=n_pre_apnea_peaks)
    pre_apnea_chest_peaks = _get_pre_event_peaks(event_start=apnea_area.start, peaks=chest_peaks, n_peaks=n_pre_apnea_peaks)
    pre_apnea_abd_baseline = np.median(np.array([abs(p.extreme_value) for p in pre_apnea_abd_peaks]))
    pre_apnea_chest_baseline = np.median(np.array([abs(p.extreme_value) for p in pre_apnea_chest_peaks]))
    # Determine the mid-event baselines for both of the signals
    apnea_abd_peaks = abd_peaks[_get_peak_index(time=apnea_area.start, peaks=abd_peaks):_get_peak_index(time=apnea_area.end, peaks=abd_peaks)]
    apnea_chest_peaks = chest_peaks[_get_peak_index(time=apnea_area.start, peaks=chest_peaks):_get_peak_index(time=apnea_area.end, peaks=chest_peaks)]
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
    # if all(b >= pre_apnea_abd_baseline * threshold_factor for b in (apnea_abd_baseline__part_1, apnea_abd_baseline__part_2)) or \
    #         all(b >= pre_apnea_chest_baseline * threshold_factor for b in (apnea_chest_baseline__part_1, apnea_chest_baseline__part_2)):
    #     return ApneaType.ObstructiveApnea
    return RespiratoryEventType.ObstructiveApnea
    # raise NotImplementedError("Could not determine ApneaType")


def detect_respiratory_events(signals: pd.DataFrame, sample_frequency_hz: float, ignore_wake_stages: bool) -> List[RespiratoryEvent]:
    assert all([col in signals for col in _NECESSARY_COLUMNS]), \
        f"At least one of the necessary columns ({_NECESSARY_COLUMNS}) is missing in the passed DataFrame"
    if ignore_wake_stages:
        assert "Is awake (ref. sleep stages)" in signals, "Seems like the dataset supports no sleep stages!"
    clusters, coarse_respiratory_event_types = _detect_airflow_resp_events(airflow_vector=signals["AIRFLOW"].values, sample_frequency_hz=sample_frequency_hz)
    chest_peaks = get_peaks(waveform=signals["CHEST"].values, filter_kernel_width=int(sample_frequency_hz*0.7))
    abd_peaks = get_peaks(waveform=signals["ABD"].values, filter_kernel_width=int(sample_frequency_hz * 0.7))

    apnea_events: List[RespiratoryEvent] = []
    n_ignored_wake_stages = 0
    n_filtered_hypopneas = 0
    for cluster, coarse_type in zip(clusters, coarse_respiratory_event_types):
        if ignore_wake_stages is True:
            is_awake = signals["Is awake (ref. sleep stages)"].values != 0
            cluster_mat = np.zeros(shape=(len(signals),), dtype=bool)
            cluster_mat[cluster.start:cluster.end] = True
            if np.sum(is_awake & cluster_mat) != 0:
                n_ignored_wake_stages += 1
                continue
        start = signals.index[cluster.start]
        end = signals.index[cluster.end]

        # If Hypopnea was detected, make sure SaO2 value falls accordingly by >= 3%
        if coarse_type == _CoarseRespiratoryEventType.Hypopnea:
            max_pre_event_sa_o2 = signals["SaO2"][start-pd.to_timedelta("30s"):start].max()
            min_post_event_sa_o2 = signals["SaO2"][start:end+pd.to_timedelta("40s")].min()
            if not min_post_event_sa_o2 <= max_pre_event_sa_o2*0.97:
                n_filtered_hypopneas += 1
                continue

        # If an apnea was detected, now further specify its type
        event_type = RespiratoryEventType.Hypopnea
        if coarse_type == _CoarseRespiratoryEventType.Apnea:
            event_type = _classify_apnea(apnea_area=cluster, abd_peaks=abd_peaks, chest_peaks=chest_peaks, sample_frequency_hz=sample_frequency_hz)
            assert event_type != RespiratoryEventType.Hypopnea, "Function _classify_apnea() must not return HypoApnea type!"

        apnea_events += [RespiratoryEvent(start=start, end=end, aux_note=None, event_type=event_type)]

    if ignore_wake_stages:
        print(f"Ignored {n_ignored_wake_stages} respiratory events, as they overlap with wake stages")
    print(f"Filtered out {n_filtered_hypopneas} hypopneas, due to SaO2 not falling by 3%")
    return apnea_events


def test_detect_respiratory_events():
    from util.datasets import SlidingWindowDataset
    from util.paths import DATA_PATH

    config = SlidingWindowDataset.Config(
        physionet_dataset_folder=DATA_PATH / "training" / "tr03-0005",
        downsample_frequency_hz=5,
        time_window_size=pd.Timedelta("2 minutes")
    )
    sliding_window_dataset = SlidingWindowDataset(config=config, allow_caching=True)

    events = detect_respiratory_events(signals=sliding_window_dataset.signals, sample_frequency_hz=config.downsample_frequency_hz, ignore_wake_stages=False)
    pass


def test_get_pre_event_peaks():
    peaks = [Peak(type=PeakType.Minimum, extreme_value=1, start=10, end=15, center=0, length=0),
             Peak(type=PeakType.Minimum, extreme_value=1, start=16, end=30, center=0, length=0),
             Peak(type=PeakType.Minimum, extreme_value=1, start=31, end=40, center=0, length=0),
             Peak(type=PeakType.Minimum, extreme_value=1, start=41, end=50, center=0, length=0),
             Peak(type=PeakType.Minimum, extreme_value=1, start=51, end=60, center=0, length=0),
             Peak(type=PeakType.Minimum, extreme_value=1, start=61, end=70, center=0, length=0)]
    assert len(_get_pre_event_peaks(event_start=5, peaks=peaks, n_peaks=5)) == 0
    assert len(_get_pre_event_peaks(event_start=12, peaks=peaks, n_peaks=5)) == 0
    assert len(_get_pre_event_peaks(event_start=15, peaks=peaks, n_peaks=5)) == 0
    assert len(_get_pre_event_peaks(event_start=16, peaks=peaks, n_peaks=5)) == 1
    assert len(_get_pre_event_peaks(event_start=30, peaks=peaks, n_peaks=5)) == 1
    assert len(_get_pre_event_peaks(event_start=100, peaks=peaks, n_peaks=5)) == 6


def test_get_peak_index():
    peaks = [Peak(type=PeakType.Minimum, extreme_value=1, start=10, end=15, center=0, length=0),
             Peak(type=PeakType.Minimum, extreme_value=1, start=16, end=30, center=0, length=0),
             Peak(type=PeakType.Minimum, extreme_value=1, start=31, end=40, center=0, length=0),
             Peak(type=PeakType.Minimum, extreme_value=1, start=41, end=50, center=0, length=0),
             Peak(type=PeakType.Minimum, extreme_value=1, start=51, end=60, center=0, length=0),
             Peak(type=PeakType.Minimum, extreme_value=1, start=61, end=70, center=0, length=0)]
    assert _get_peak_index(time=33, peaks=peaks) == 2
    assert _get_peak_index(time=31, peaks=peaks) == 2
    assert _get_peak_index(time=40, peaks=peaks) == 2
    assert _get_peak_index(time=41, peaks=peaks) == 3
    assert _get_peak_index(time=9, peaks=peaks) is None
    assert _get_peak_index(time=71, peaks=peaks) is None


def test_detect_airflow_resp_events__jit_speed():
    from datetime import datetime
    import os.path

    sample_vector = np.load(file=os.path.dirname(os.path.realpath(__file__)) + "/sample_airflow_signal_10hz.npy")
    _detect_airflow_resp_events(airflow_vector=sample_vector, sample_frequency_hz=10)  # One initial run to JIT the code

    n_runs = 1000
    started_at = datetime.now()
    for n in range(n_runs):
        result = _detect_airflow_resp_events(airflow_vector=sample_vector, sample_frequency_hz=10)
    overall_seconds = (datetime.now()-started_at).total_seconds()

    print()
    print(f"The whole process with n_runs={n_runs} took {overall_seconds*1000:.1f}ms")
    print(f"A single run took {overall_seconds/n_runs*1000:.2f}ms")
