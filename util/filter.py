from typing import Union
from functools import lru_cache

import scipy.signal
import numpy as np
import pandas as pd


__author__ = "Robert Voelckner"
__copyright__ = "Copyright 2021"
__license__ = "MIT"


@lru_cache(maxsize=None)
def _get_butterworth_bandpass_coefficients(f_low_cutoff: float, f_high_cutoff: float, f_sample: float, filter_order=5):
    f_nyq = 0.5 * f_sample
    low = f_low_cutoff / f_nyq
    high = f_high_cutoff / f_nyq
    sos_coeff = scipy.signal.butter(N=filter_order, Wn=[low, high], analog=False, btype='band', output="sos")
    return sos_coeff


@lru_cache(maxsize=None)
def _get_butterworth_lowpass_coefficients(f_cutoff: float, f_sample: float, filter_order: int = 5):
    f_nyq = 0.5 * f_sample
    cut = f_cutoff / f_nyq
    sos_coeff = scipy.signal.butter(N=filter_order, Wn=cut, analog=False, btype='low', output="sos")
    return sos_coeff


def apply_butterworth_bandpass_filter(data: Union[np.ndarray, pd.Series], f_low_cutoff: float, f_high_cutoff: float,
                                      f_sample: float, filter_order: int = 5) -> Union[np.ndarray, pd.Series]:
    sos_coeff = _get_butterworth_bandpass_coefficients(f_low_cutoff, f_high_cutoff, f_sample, filter_order=filter_order)
    y = scipy.signal.sosfiltfilt(sos_coeff, data.values if isinstance(data, pd.Series) else data)
    assert np.sum(np.isnan(y)) == 0, \
        "Filter output contains at least one NaN. That's not desired. Try lowering filter-order parameter."
    if isinstance(data, pd.Series):
        y = pd.Series(data=y, index=data.index, name=f"{data.name} (Butterworth bp, {filter_order} order)")
    return y


def apply_butterworth_lowpass_filter(data: Union[np.ndarray, pd.Series], f_cutoff: float, f_sample: float,
                                     filter_order: int = 5) -> Union[np.ndarray, pd.Series]:
    sos_coeff = _get_butterworth_lowpass_coefficients(f_cutoff, f_sample, filter_order=filter_order)
    y = scipy.signal.sosfiltfilt(sos_coeff, data.values if isinstance(data, pd.Series) else data)
    assert np.sum(np.isnan(y)) == 0, \
        "Filter output contains at least one NaN. That's not desired. Try lowering filter-order parameter."
    if isinstance(data, pd.Series):
        y = pd.Series(data=y, index=data.index, name=f"{data.name} (Butterworth lp, {filter_order} order)")
    return y


def test_butterworth_bandpass_response():
    # The following code was inspired by   https://stackoverflow.com/a/12233959
    import matplotlib.pyplot as plt

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 5000.0
    lowcut = 500.0
    highcut = 700.0

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        sos = _get_butterworth_bandpass_coefficients(lowcut, highcut, fs, filter_order=order)
        # sos = _get_butterworth_lowpass_coefficients(highcut, fs, filter_order=order)
        w, h = scipy.signal.sosfreqz(sos, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    # Filter a noisy signal.
    T = 0.20
    nsamples = int(T * fs)
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = apply_butterworth_bandpass_filter(x, lowcut, highcut, fs, filter_order=6)
    # y = apply_butterworth_lowpass_filter(x, highcut, fs, filter_order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()
