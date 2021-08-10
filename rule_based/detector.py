from typing import List

import pandas as pd
import numpy as np
import numba

from util.mathutil import PeakType, Cluster, get_peaks, cluster_1d


__author__ = "Robert Voelckner"
__copyright__ = "Copyright 2021"
__license__ = "MIT"


_NECESSARY_COLUMNS = ("AIRFLOW", "ABD", "CHEST", "SaO2")


@numba.jit(nopython=True)
def _detect_potential_apnea_airflow_areas(airflow_vector: np.ndarray, sample_frequency_hz: float) -> List[Cluster]:
    filter_kernel_width = int(sample_frequency_hz*0.7)
    peaks = get_peaks(waveform=airflow_vector, filter_kernel_width=filter_kernel_width)

    # Figure out potential low-amplitude areas
    overall_baseline = np.median(np.array([abs(p.extreme_value) for p in peaks]))
    low_peaks = [p for p in peaks if abs(p.extreme_value) <= overall_baseline]
    low_peaks_mat = np.zeros(shape=airflow_vector.shape)
    for p in low_peaks:
        low_peaks_mat[p.start:p.end] = p.extreme_value
    low_peak_clusters = cluster_1d(low_peaks_mat, no_klass=0, allowed_distance=int(sample_frequency_hz*0.5), min_length=int(sample_frequency_hz * 10))

    # Now, do some further filtering with the low-amplitude areas we found
    potential_apnea_clusters: List[Cluster] = []
    for cluster in low_peak_clusters:
        cluster_peaks = [p for p in peaks if cluster.start <= p.center <= cluster.end]
        n_cluster_peaks_below_baseline = len([p for p in cluster_peaks if abs(p.extreme_value) <= overall_baseline])
        if n_cluster_peaks_below_baseline / len(cluster_peaks) >= 0.8:
            potential_apnea_clusters.append(cluster)

    return potential_apnea_clusters


def detect_apneas(window_signals: pd.DataFrame, sample_frequency_hz: float):
    assert all([col in window_signals for col in _NECESSARY_COLUMNS]), \
        f"At least one of the necessary columns ({_NECESSARY_COLUMNS}) is not included in the passed DataFrame"
    pass


def test__detect_potential_apnea_airflow_areas():
    sample_vector = (np.random.rand(1000)*10) - 5
    clusters = _detect_potential_apnea_airflow_areas(airflow_vector=sample_vector, sample_frequency_hz=10)
    pass
