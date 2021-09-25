import math
import numpy
from typing import List, NamedTuple, Optional
from enum import Enum

import numpy as np
import numba


class PeakType(Enum):
    Minimum = 0
    Maximum = 1


class ZeroCrossType(Enum):
    Positive = 0
    Negative = 1


Peak = NamedTuple("Peak", type=PeakType, extreme_value=float, start=int, end=int, center=int, length=int)
ZeroCross = NamedTuple("ZeroCross", type=ZeroCrossType, position=int)
IntRange = NamedTuple("IntRange", start=int, end=int, length=int)


@numba.jit(nopython=True)
def get_peaks(waveform: np.ndarray, filter_kernel_width: int) -> List[Peak]:
    """
    Detects min/max peaks of a around-zero centered waveform.

    :param waveform: Input waveform. Must be centered around zero.
    :param filter_kernel_width: Width of the filter kernel. Most likely the number
                                of samples that form one signal period.
    :return: A list of the detected peaks.
    """
    # Convolve waveform with the filter kernel & cut the right and left overlaps
    filter_kernel = np.ones(filter_kernel_width) / filter_kernel_width
    filtered_waveform = np.convolve(waveform, filter_kernel)
    filtered_waveform = filtered_waveform[int(filter_kernel_width/2) - 1:-math.ceil(filter_kernel_width/2)]
    if len(filtered_waveform) != len(waveform):
        raise AssertionError(f"There is a length mismatch in filtered and original waveform. Needs some rework!")

    # Align our filtered waveform with the original signal
    shift = int(-filter_kernel_width / 4)
    if shift != 0:
        filtered_waveform = np.roll(filtered_waveform, shift=shift)
        filtered_waveform[shift:] = np.nan

    zero_crosses_pos: np.ndarray = np.where(np.diff(np.sign(filtered_waveform)) > 1)[0]
    zero_crosses_neg: np.ndarray = np.where(np.diff(np.sign(filtered_waveform)) < -1)[0]
    if np.abs(len(zero_crosses_pos)-len(zero_crosses_neg)) > 1:
        raise AssertionError("There's a discrepancy in numbers of detected pos/neg zero crosses. Needs some rework!")

    peaks: List[Peak] = []
    if len(zero_crosses_pos) == 0 or len(zero_crosses_neg) == 0:
        return peaks

    # Merge our zero crosses into one list & do some sanity checks
    zero_crosses: List[ZeroCross] = []
    last_zero_cross_type: Optional[ZeroCrossType] = None
    while len(zero_crosses_pos) != 0 or len(zero_crosses_neg) != 0:
        position_pos = zero_crosses_pos[0] if len(zero_crosses_pos) != 0 else None
        position_neg = zero_crosses_neg[0] if len(zero_crosses_neg) != 0 else None
        if position_neg is None or (position_pos is not None and position_pos < position_neg):
            if last_zero_cross_type is not None and last_zero_cross_type == ZeroCrossType.Positive:
                raise AssertionError("Consecutive positive zero crosses of the same type are not supported!")
            zero_crosses.append(ZeroCross(type=ZeroCrossType.Positive, position=position_pos))
            zero_crosses_pos = zero_crosses_pos[1:]
            last_zero_cross_type = ZeroCrossType.Positive
        else:
            if last_zero_cross_type is not None and last_zero_cross_type == ZeroCrossType.Negative:
                raise AssertionError("Consecutive negative zero crosses of the same type are not supported!")
            zero_crosses.append(ZeroCross(type=ZeroCrossType.Negative, position=position_neg))
            zero_crosses_neg = zero_crosses_neg[1:]
            last_zero_cross_type = ZeroCrossType.Negative

    # Now, let's create the list of peaks
    last_zero_cross: Optional[ZeroCross] = None
    for zero_cross in zero_crosses:
        if last_zero_cross is None:
            last_zero_cross = zero_cross
            continue
        if zero_cross.type == ZeroCrossType.Positive:
            extreme_value = np.min(waveform[last_zero_cross.position:zero_cross.position])
            peak_type = PeakType.Minimum
        else:
            extreme_value = np.max(waveform[last_zero_cross.position:zero_cross.position])
            peak_type = PeakType.Maximum
        start: int = last_zero_cross.position
        end: int = zero_cross.position-1
        peaks.append(Peak(type=peak_type, extreme_value=extreme_value, start=start, end=end, center=int(start+(end-start)/2), length=end-start+1))
        last_zero_cross = zero_cross
    return peaks


@numba.jit(nopython=True)
def cluster_1d(input_vector: np.ndarray, no_klass: int = 0, allowed_distance: int = 1, min_length: int = 5) -> List[IntRange]:
    klass_positions: np.ndarray = np.where(input_vector != no_klass)[0]
    clusters: List[IntRange] = []
    cluster__last_valid_position: Optional[int] = None
    cluster__start: Optional[int] = None

    for position in klass_positions:
        # klass = input_vector[position]
        if cluster__last_valid_position is None:
            cluster__start = cluster__last_valid_position = position
            continue
        if (position-cluster__last_valid_position-1) > allowed_distance:
            cluster_length = cluster__last_valid_position-cluster__start+1
            if cluster_length >= min_length:
                clusters.append(IntRange(start=cluster__start, end=cluster__last_valid_position, length=cluster_length))
            cluster__start = position
        cluster__last_valid_position = position

    # Perhaps, there's yet another cluster in the very last position. Let's check
    cluster_length = cluster__last_valid_position - cluster__start + 1
    if cluster_length >= min_length:
        clusters.append(IntRange(start=cluster__start, end=cluster__last_valid_position, length=cluster_length))
    return clusters


@numba.jit(nopython=True)
def normalize_robust(input: np.ndarray) -> np.ndarray:
    """
    Normalizes an input signal to:
    - median = 0
    - Interquartile range = 1 (range between 25th and 75th percentile)

    @note
    This way of normalizing is much more robust towards outliers than normalization using mean and std-dev.
    """
    input_above_threshold = input[np.abs(input) >= 1e-10]  # That filter is necessary to be able to deal with "kaputt" data
    if len(input_above_threshold) == 0:
        return input.astype(numpy.float32)
    median = np.median(input_above_threshold)
    quartiles = np.quantile(input_above_threshold, q=(0.75, 0.25))
    inter_quartile_range = quartiles[0] - quartiles[1]
    if inter_quartile_range <= 1e-4:  # Important to avoid division by zero & unrealistic upscaling due to "kaputt" data
        inter_quartile_range = 1
    return ((input-median)/inter_quartile_range).astype(numpy.float32)


def test_normalize_robust():
    def inter_quartile_range(x): return np.quantile(x, 0.75) - np.quantile(x, 0.25)

    x = np.array([185.24931, 142.84528, 97.157455, 49.803917, 24.945267, 1.1902198, -53.66841, -117.8662, -216.58253, -361.31833, -500.33585, -595.6547, -654.03674, -674.5523, -663.14276, -630.9965])
    assert not np.isclose(inter_quartile_range(x), 1, atol=0.001)
    assert not np.isclose(np.median(x), 0, atol=0.001)

    y = normalize_robust(x)
    assert np.isclose(inter_quartile_range(y), 1, atol=0.001)
    assert np.isclose(np.median(y), 0, atol=0.001)


def test_normalize_robust__equal_input_values():
    x = np.array([100] * 10)
    y = normalize_robust(x)
    assert not np.any(np.isnan(y))
    assert np.allclose(y, 0.0)


def test_cluster_1d_1():
    input_vector = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1])
    clusters = cluster_1d(input_vector=input_vector, no_klass=0, allowed_distance=1, min_length=5)
    assert len(clusters) == 2
    assert IntRange(start=2, end=8, length=7) in clusters
    assert IntRange(start=11, end=17, length=7) in clusters


def test_cluster_1d_2():
    input_vector = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1])
    clusters = cluster_1d(input_vector=input_vector, no_klass=0, allowed_distance=1, min_length=5)
    assert len(clusters) == 1
    assert IntRange(start=2, end=8, length=7) in clusters


def test_cluster_1d_3():
    input_vector = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1])
    clusters = cluster_1d(input_vector=input_vector, no_klass=0, allowed_distance=0, min_length=5)
    assert len(clusters) == 0


def test_cluster_1d_4():
    input_vector = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1])
    clusters = cluster_1d(input_vector=input_vector, no_klass=0, allowed_distance=0, min_length=4)
    assert len(clusters) == 2
    assert IntRange(start=8, end=12, length=5) in clusters
    assert IntRange(start=14, end=17, length=4) in clusters


def test_cluster_1d__jit_speed():
    from datetime import datetime

    sample_signal = np.random.rand(1000).round(decimals=0).astype(int)
    cluster_1d(input_vector=sample_signal)  # One initial run to JIT the code

    n_runs = 5000
    started_at = datetime.now()
    for n in range(n_runs):
        result = cluster_1d(input_vector=sample_signal, no_klass=0, allowed_distance=1, min_length=3)
    overall_seconds = (datetime.now()-started_at).total_seconds()

    print()
    print(f"The whole process with n_runs={n_runs} took {overall_seconds*1000:.1f}ms")
    print(f"A single run took {overall_seconds/n_runs*1000:.2f}ms")


def test_get_peaks__qualitative():
    import matplotlib.pyplot as plt
    T = 4*np.pi
    f_s = 10
    x = np.arange(T*f_s)
    y = np.sin(x/f_s)

    # --------------------- Make some dev-helping plots. Code was (almost) entirely taken from the function above.
    filter_size = int(f_s)
    # filter = np.append(-np.ones(filter_size), np.ones(filter_size)) / filter_size
    filter = np.ones(filter_size) / filter_size
    filtered_waveform = np.convolve(y, filter)
    filtered_waveform = filtered_waveform[int(filter_size/2) - 1:-math.ceil(filter_size/2)]
    if len(filtered_waveform) != len(y):
        raise AssertionError(f"There is a length mismatch in filtered and original waveform. Needs some rework!")

    # Align our filtered waveform with the original signal
    shift = int(-filter_size/4)
    if shift != 0:
        filtered_waveform = np.roll(filtered_waveform, shift=shift)
        filtered_waveform[shift:] = np.nan

    plt.plot(y)
    plt.plot(filtered_waveform)
    plt.show()
    # --------------

    peaks = get_peaks(waveform=y, filter_kernel_width=int(f_s))

    assert len(peaks) == 2
    assert peaks[0].type == PeakType.Minimum
    assert peaks[1].type == PeakType.Maximum
    pass


def test_get_peaks__jit_speed():
    from datetime import datetime
    from .paths import UTIL_PATH

    sample_signal = np.load(file=UTIL_PATH/"sample_airflow_signal_10hz.npy")
    get_peaks(waveform=sample_signal, filter_kernel_width=5)  # One initial run to JIT the code

    n_runs = 5000
    started_at = datetime.now()
    for n in range(n_runs):
        result = get_peaks(waveform=sample_signal, filter_kernel_width=50)
    overall_seconds = (datetime.now()-started_at).total_seconds()

    print()
    print(f"The whole process with n_runs={n_runs} took {overall_seconds*1000:.1f}ms")
    print(f"A single run took {overall_seconds/n_runs*1000:.2f}ms")


def test_get_peaks__example_plot():
    import pandas as pd
    import matplotlib.pyplot as plt
    from .paths import UTIL_PATH

    sample_signal = np.load(file=UTIL_PATH/"sample_airflow_signal_10hz.npy")
    peaks = get_peaks(waveform=sample_signal, filter_kernel_width=5)
    peaks_mat = np.zeros(shape=(sample_signal.shape[0],))
    for p in peaks:
        peaks_mat[p.center] = p.extreme_value

    sample_signal_series = pd.Series(sample_signal)
    peaks_series = pd.Series(peaks_mat)
    df = pd.concat([sample_signal_series, peaks_series], axis=1)

    df.plot(figsize=(15, 6), subplots=False)
    plt.show()
