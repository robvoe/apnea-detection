import numpy as np
import pandas as pd

from util.mathutil import Peak, PeakType
from .detector import detect_respiratory_events, _get_pre_event_peaks, _get_peak_index, _detect_airflow_resp_events


def test_get_pre_event_peaks():
    peaks = [Peak(type=PeakType.Minimum, extreme_value=1, start=10, end=15, center=0, length=0),
             Peak(type=PeakType.Minimum, extreme_value=1, start=16, end=30, center=0, length=0),
             Peak(type=PeakType.Minimum, extreme_value=1, start=31, end=40, center=0, length=0),
             Peak(type=PeakType.Minimum, extreme_value=1, start=41, end=50, center=0, length=0),
             Peak(type=PeakType.Minimum, extreme_value=1, start=51, end=60, center=0, length=0),
             Peak(type=PeakType.Minimum, extreme_value=1, start=61, end=70, center=0, length=0)]
    assert len(_get_pre_event_peaks(event_start_time=5, peaks=peaks, n_peaks=5)) == 0
    assert len(_get_pre_event_peaks(event_start_time=12, peaks=peaks, n_peaks=5)) == 0
    assert len(_get_pre_event_peaks(event_start_time=15, peaks=peaks, n_peaks=5)) == 0
    assert len(_get_pre_event_peaks(event_start_time=16, peaks=peaks, n_peaks=5)) == 1
    assert len(_get_pre_event_peaks(event_start_time=30, peaks=peaks, n_peaks=5)) == 1
    assert len(_get_pre_event_peaks(event_start_time=100, peaks=peaks, n_peaks=5)) == 6


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
    from util.paths import UTIL_PATH

    sample_vector = np.load(file=UTIL_PATH/"sample_airflow_signal_10hz.npy")
    _detect_airflow_resp_events(airflow_vector=sample_vector, sample_frequency_hz=10)  # One initial run to JIT the code

    n_runs = 1000
    started_at = datetime.now()
    for n in range(n_runs):
        result = _detect_airflow_resp_events(airflow_vector=sample_vector, sample_frequency_hz=10)
    overall_seconds = (datetime.now()-started_at).total_seconds()

    print()
    print(f"The whole process with n_runs={n_runs} took {overall_seconds*1000:.1f}ms")
    print(f"A single run took {overall_seconds/n_runs*1000:.2f}ms")