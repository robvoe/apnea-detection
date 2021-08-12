from typing import List, Tuple
from enum import Enum

import pandas as pd
import numpy as np
import numba
import numba.typed

from util.mathutil import PeakType, Cluster, get_peaks, cluster_1d, Peak
from .datasets import ApneaEvent, ApneaType

__author__ = "Robert Voelckner"
__copyright__ = "Copyright 2021"
__license__ = "MIT"


_NECESSARY_COLUMNS = ("AIRFLOW", "ABD", "CHEST", "SaO2")


class _CoarseApneaType(Enum):
    Apnea = 0
    HypoApnea = 1


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
def _detect_airflow_apnea_areas(airflow_vector: np.ndarray, sample_frequency_hz: float) -> Tuple[List[Cluster], List[_CoarseApneaType]]:
    min_apnea_pulse_length = 10*sample_frequency_hz
    moving_baseline_window_lr = 100  # specifies each direction (left/right) from current peak_index-position
    filter_kernel_width = int(sample_frequency_hz*0.7)
    n_reference_peaks = 3
    peaks: List[Peak] = get_peaks(waveform=airflow_vector, filter_kernel_width=filter_kernel_width)

    apnea_areas: List[Cluster] = []
    coarse_apnea_types: List[_CoarseApneaType] = []
    peak_index = 0
    while peak_index < len(peaks)-n_reference_peaks:
        # Determine a moving baseline for values that surround our current peak_index-position
        moving_window_left_index = max(peak_index - moving_baseline_window_lr, 0)
        moving_window_right_index = min(peak_index + moving_baseline_window_lr, len(peaks))
        window_peaks = peaks[moving_window_left_index:moving_window_right_index]
        moving_baseline = np.median(np.array([abs(p.extreme_value) for p in window_peaks]))
        max_allowed_moving_baseline_value = 1.0 * moving_baseline

        # Determine the reference-peaks-baseline defined by the peaks directly at our current peak_index-position
        reference_peaks = peaks[peak_index:peak_index + n_reference_peaks]
        reference_peaks_baseline = np.sqrt(np.mean(np.array([np.square(abs(p.extreme_value)) for p in reference_peaks])))
        # Determine how many subsequent peaks we need to cover at least 8..10s
        head_index = peak_index + n_reference_peaks
        lengths_right = np.array([p.length for p in peaks[head_index:]]).cumsum()
        cumulated_lengths_larger_than = (lengths_right >= min_apnea_pulse_length)
        if np.sum(cumulated_lengths_larger_than) == 0:
            break
        tail_index = head_index + cumulated_lengths_larger_than.argmax()
        # Try to stretch the window longer, whilst preserving its baseline smaller than our baselines
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
        # Tail is stretched. Now make sure the signal rises up to its initial (high) value afterwards again
        post_tail_max_peak_index = min(len(peaks), tail_index+10)
        post_tail_peaks = peaks[tail_index:post_tail_max_peak_index]
        if np.sum(np.array([abs(p.extreme_value) > reference_peaks_baseline*0.9 for p in post_tail_peaks])) == 0:
            peak_index += 1
            continue
        # Now, let the beginning of the window reach to the most-negative dip right before
        pre_head_peaks = peaks[head_index-1:head_index+2]
        min_index = np.argmin(np.array([p.extreme_value for p in pre_head_peaks]))
        most_negative_dip_index = head_index - 1 + min_index
        # Determine the coarse type of our apnea:  Apnea/HypoApnea
        window_peaks = peaks[head_index:tail_index + 1]
        window_baseline = np.percentile(np.array([abs(p.extreme_value) for p in window_peaks]), 25)
        type_decision_reference_peaks_baseline = np.max(np.array([(abs(p.extreme_value)) for p in reference_peaks]))
        coarse_apnea_type: _CoarseApneaType = _CoarseApneaType.Apnea if window_baseline <= 0.1 * type_decision_reference_peaks_baseline else _CoarseApneaType.HypoApnea
        coarse_apnea_types.append(coarse_apnea_type)

        start = peaks[most_negative_dip_index].center
        end = peaks[tail_index].end
        apnea_areas.append(Cluster(start=start, end=end, length=end-start+1))

        peak_index = tail_index
    return apnea_areas, coarse_apnea_types


def detect_apneas(signals: pd.DataFrame, sample_frequency_hz: float) -> List[ApneaEvent]:
    assert all([col in signals for col in _NECESSARY_COLUMNS]), \
        f"At least one of the necessary columns ({_NECESSARY_COLUMNS}) is missing in the passed DataFrame"
    clusters, coarse_apnea_types = _detect_airflow_apnea_areas(airflow_vector=signals["AIRFLOW"].values, sample_frequency_hz=sample_frequency_hz)

    apnea_events: List[ApneaEvent] = []
    for cluster, coarse_apnea_type in zip(clusters, coarse_apnea_types):
        start = signals.index[cluster.start]
        end = signals.index[cluster.end]
        apnea_type = ApneaType.HypoApnea if coarse_apnea_type == _CoarseApneaType.HypoApnea else ApneaType.CentralApnea
        # TODO
        apnea_events += [ApneaEvent(start=start, end=end, aux_note=None, apnea_type=apnea_type)]

    return apnea_events


def test_detect_apneas():
    from util.datasets import SlidingWindowDataset
    from util.paths import DATA_PATH

    config = SlidingWindowDataset.Config(
        physionet_dataset_folder=DATA_PATH / "training" / "tr03-0005",
        downsample_frequency_hz=5,
        time_window_size=pd.Timedelta("2 minutes")
    )
    sliding_window_dataset = SlidingWindowDataset(config=config, allow_caching=True)

    apneas = detect_apneas(signals=sliding_window_dataset.signals, sample_frequency_hz=config.downsample_frequency_hz)
    pass


def test_detect_airflow_apnea_areas__jit_speed():
    from datetime import datetime
    import os.path

    sample_vector = np.load(file=os.path.dirname(os.path.realpath(__file__)) + "/sample_airflow_signal_10hz.npy")
    _detect_airflow_apnea_areas(airflow_vector=sample_vector, sample_frequency_hz=10)  # One initial run to JIT the code

    n_runs = 1000
    started_at = datetime.now()
    for n in range(n_runs):
        result = _detect_airflow_apnea_areas(airflow_vector=sample_vector, sample_frequency_hz=10)
    overall_seconds = (datetime.now()-started_at).total_seconds()

    print()
    print(f"The whole process with n_runs={n_runs} took {overall_seconds*1000:.1f}ms")
    print(f"A single run took {overall_seconds/n_runs*1000:.2f}ms")
