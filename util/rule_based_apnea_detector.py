from typing import List

import pandas as pd
import numpy as np
import numba

from util.mathutil import PeakType, Cluster, get_peaks, cluster_1d, Peak

__author__ = "Robert Voelckner"
__copyright__ = "Copyright 2021"
__license__ = "MIT"


_NECESSARY_COLUMNS = ("AIRFLOW", "ABD", "CHEST", "SaO2")


# @numba.jit(nopython=True)
def _detect_potential_apnea_airflow_areas(airflow_vector: np.ndarray, sample_frequency_hz: float) -> List[Cluster]:
    filter_kernel_width = int(sample_frequency_hz*0.7)
    peaks = get_peaks(waveform=airflow_vector, filter_kernel_width=filter_kernel_width)

    # Figure out potential low-amplitude areas
    overall_baseline = np.median(np.array([abs(p.extreme_value) for p in peaks]))
    low_peaks = [p for p in peaks if abs(p.extreme_value) <= overall_baseline*0.9]
    low_peaks_mat = np.zeros(shape=airflow_vector.shape)
    for p in low_peaks:
        low_peaks_mat[p.start:p.end] = p.extreme_value
    low_peak_clusters = cluster_1d(low_peaks_mat, no_klass=0, allowed_distance=int(sample_frequency_hz*0.5), min_length=int(sample_frequency_hz * 7))

    # Now, do some further filtering with the low-amplitude areas we found
    # potential_apnea_clusters: List[Cluster] = []
    # for cluster in low_peak_clusters:
    #     cluster_peaks = [p for p in peaks if cluster.start <= p.center <= cluster.end]
    #     n_cluster_peaks_below_baseline = len([p for p in cluster_peaks if abs(p.extreme_value) <= overall_baseline])
    #     if n_cluster_peaks_below_baseline / len(cluster_peaks) >= 0.8:
    #         potential_apnea_clusters.append(cluster)

    # Find out, which of the detected low-amplitude areas are apneas
    apnea_areas: List[Cluster] = []
    for cluster in low_peak_clusters:
        cluster_peaks = [p for p in peaks if cluster.start <= p.center <= cluster.end]
        pre_cluster_peaks = [p for p in peaks if p.center < cluster.start]
        if len(pre_cluster_peaks) == 0:
            continue
        n_pre_cluster_peaks = int(min(len(pre_cluster_peaks), 3))
        pre_cluster_peaks = pre_cluster_peaks[-n_pre_cluster_peaks:]

        cluster_baseline = np.median(np.array([abs(p.extreme_value) for p in cluster_peaks]))
        if any([abs(p.extreme_value*0.3) >= cluster_baseline for p in pre_cluster_peaks]):
            apnea_areas.append(cluster)
    return apnea_areas


# @numba.jit(nopython=True)
def _detect_airflow_apnea_areas__new(airflow_vector: np.ndarray, sample_frequency_hz: float) -> List[Cluster]:
    min_apnea_pulse_length = 8*sample_frequency_hz
    filter_kernel_width = int(sample_frequency_hz*0.7)
    peaks = get_peaks(waveform=airflow_vector, filter_kernel_width=filter_kernel_width)
    # overall_baseline = np.sqrt(np.mean([np.square(p.extreme_value) for p in peaks]))
    overall_baseline = np.median([abs(p.extreme_value) for p in peaks])

    apnea_areas: List[Cluster] = []
    while len(peaks) > 2:
        reference_peaks = peaks[0:2]
        reference_peaks_baseline = np.sqrt(np.mean(np.array([np.square(abs(p.extreme_value)) for p in reference_peaks])))
        # Determine how many peaks we need to cover at least 10s
        head_index = 2
        tail_index = head_index
        for tail_index in range(tail_index, len(peaks[1:])):
            if np.sum(np.array([p.length for p in peaks[head_index:tail_index+1]])) >= min_apnea_pulse_length:
                break
        if np.sum(np.array([p.length for p in peaks[head_index:tail_index+1]])) < min_apnea_pulse_length:
            peaks = peaks[1:]
            continue
        # Try to stretch the window longer, whilst preserving its baseline smaller than our baselines
        window_baseline = np.max(np.array([abs(p.extreme_value) for p in peaks[head_index:tail_index + 1]]))
        if window_baseline > reference_peaks_baseline * 0.7 or window_baseline > overall_baseline:
            peaks = peaks[1:]
            continue
        for tail_index in range(tail_index+1, len(peaks[1:])):
            window_baseline = np.max(np.array([abs(p.extreme_value) for p in peaks[head_index:tail_index + 1]]))
            if window_baseline > reference_peaks_baseline * 0.7 or window_baseline > overall_baseline:
                tail_index -= 1
                break
        # Tail is stretched. Now make sure the signal rises up to its initial (high) value afterwards again
        post_tail_max_peak_index = min(len(peaks), tail_index+10)
        post_tail_peaks = peaks[tail_index:post_tail_max_peak_index]
        if not any([abs(p.extreme_value) > reference_peaks_baseline*0.9 for p in post_tail_peaks]):
            peaks = peaks[1:]
            continue

        start = peaks[head_index].start
        end = peaks[tail_index].end
        apnea_areas.append(Cluster(start=start, end=end, length=end-start+1))

        peaks = peaks[tail_index:]
        # peaks = peaks[2:]
    return apnea_areas


def detect_apneas(window_signals: pd.DataFrame, sample_frequency_hz: float):
    assert all([col in window_signals for col in _NECESSARY_COLUMNS]), \
        f"At least one of the necessary columns ({_NECESSARY_COLUMNS}) is not included in the passed DataFrame"
    pass


def test__detect_potential_apnea_airflow_areas():
    import os.path
    sample_vector = np.load(file=os.path.dirname(os.path.realpath(__file__)) + "/sample_signal.npy")

    # clusters = _detect_potential_apnea_airflow_areas(airflow_vector=sample_vector, sample_frequency_hz=10)
    clusters = _detect_airflow_apnea_areas__new(airflow_vector=sample_vector, sample_frequency_hz=10)
    pass
