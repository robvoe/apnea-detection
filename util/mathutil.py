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


Peak = NamedTuple("Peak", type=PeakType, extreme_value=float, start=int, end=int)
ZeroCross = NamedTuple("ZeroCross", type=ZeroCrossType, position=int)


@numba.jit(nopython=True)
def get_peaks(waveform: np.ndarray, filter_kernel_width: int) -> List[Peak]:
    """
    Detects min/max peaks of a around-zero centered waveform.

    :param waveform: Input waveform. Must be centered around zero.
    :param filter_kernel_width: Width of the filter kernel. Most likely the number
                                of samples that form one signal period.
    :return:
    """
    filter = np.append(-np.ones(filter_kernel_width), np.ones(filter_kernel_width))/filter_kernel_width

    # Convolve waveform with the filter kernel & cut the right and left overlaps
    filtered_waveform = np.convolve(waveform, filter)
    filtered_waveform = filtered_waveform[filter_kernel_width-1:-filter_kernel_width]
    if len(filtered_waveform) != len(waveform):
        raise AssertionError(f"There is a length mismatch in filtered and original waveform. Needs some rework!")

    # Align our filtered waveform with the original signal
    shift = -int(np.pi/2 * filter_kernel_width)
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
        peaks.append(Peak(type=peak_type, extreme_value=extreme_value, start=last_zero_cross.position, end=zero_cross.position-1))
        last_zero_cross = zero_cross
    return peaks


def test_1():
    import matplotlib.pyplot as plt
    T = 4*np.pi
    f_s = 10
    x = np.arange(T*f_s)
    y = np.sin(x/f_s)

    # --------------------- Make some helping plots
    filter_size = int(f_s)
    # filter = np.append(-np.ones(filter_size), np.ones(filter_size)) / filter_size
    filter = np.ones(filter_size) / filter_size
    filtered_waveform = np.convolve(y, filter)
    filtered_waveform = filtered_waveform[int(filter_size/2) - 1:-int(filter_size/2)]
    if len(filtered_waveform) != len(y):
        raise AssertionError(f"There is a length mismatch in filtered and original waveform. Needs some rework!")

    # Align our filtered waveform with the original signal
    shift = int(-filter_size/2)
    filtered_waveform = np.roll(filtered_waveform, shift=shift)
    filtered_waveform[shift:] = np.nan

    plt.plot(y)
    plt.plot(filtered_waveform)
    plt.show()
    # --------------

    peaks = get_peaks(waveform=y, filter_kernel_width=int(f_s))

    assert len(peaks) == 3
    assert peaks[0].type == PeakType.Maximum
    assert peaks[1].type == PeakType.Minimum
    assert peaks[2].type == PeakType.Maximum
    pass
