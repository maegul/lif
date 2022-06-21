"""Estimate the pure sinusoidal amplitude necessary to lead to an empirically
reported F1 which is affected by the combination of rectification and DC shift
"""

from typing import Tuple, Callable, Optional

from functools import partial

import numpy as np
from numpy import fft  # type: ignore

import scipy.optimize as opt
from scipy.optimize import minimize

from ..utils.units.units import TempFrequency, Time, val_gen


def gen_sin(
        amplitude: float, DC_amp: float,
        time: Time[val_gen],
        freq: TempFrequency[float] = TempFrequency(1)) -> val_gen:
    """Generate Sinusoid of frequency freq over time

    If to be used for deriving real amplitude from empirical F1,
    time must be such that size*temp_res = 1

    Parameters
    ----
    time:
        Can be either array or float
    DC_amp:
        Constant amplitude added to whole sinusoid
        Intended to represent the mean firing rate of a neuron

    Returns
    ----

    """

    signal = amplitude * np.sin(freq.hz * 2*np.pi*time.s) + DC_amp  # type: ignore
    signal: val_gen

    return signal


def gen_fft(
        signal: np.ndarray, time: Time[np.ndarray],
        view_max=20) -> Tuple[np.ndarray, np.ndarray]:
    """Generate FFT of signal over time, returning spectrum and freqs

    FFT is normalised to the size of the signal.

    view_max: number of items returned from beginning for both
        spectrum and frequency

    Returns
    -------
    spectrum, frequencies
    """

    signal_size = time.value.size

    spec = fft.rfft(signal)

    spec[1:] *= 2/signal_size  # to normalise the split / coonjugate values
    spec[0] /= signal_size
    freq = fft.rfftfreq(signal_size, (time.s[1]-time.s[0]))

    return spec[:view_max], freq[:view_max]


def amp_f1_diff(
        amplitude: float, DC_amp: float,
        time: Time[np.ndarray],
        f1_target: float, freq: TempFrequency[float] = TempFrequency(1)):
    """Find diff between target f1 and f1 of rectified sin defined by amplitude and DC_amp

    freq must be an integer and will be cast as such, as it is used to index into an FFT.
    The FFT is made such that at indices 0, 1, 2, ... are freqs 0, 1, 2, ... .
    """

    signal = gen_sin(amplitude, DC_amp, time, freq=freq)
    signal[signal < 0] = 0

    spec, _ = gen_fft(signal=signal, time=time)

    freq_idx = int(freq.hz)
    current_f1 = np.abs(spec[freq_idx])

    return np.abs(current_f1 - f1_target)


def opt_f1(
        time: Time[np.ndarray], c: float,
        f1_target: float, f: TempFrequency = TempFrequency(1)) -> Callable:
    """Return partial of amp_f1_diff that takes only amplitude and DC_amp for optimisation"""

    # freq workings relies on t being 1 second with even increments/period
    # so that t.size * t[1]-t[0] is 1
    # this ensures that the freqs in the spec are 0, 1, 2, ... n/2, as
    # time_size*time_res is the denominating factor in the freqs series (see np.fft.rfftfreq docs)
    time_size = time.value.size
    time_res = (time.s[1] - time.s[0])
    t_test = time_size * time_res

    if not t_test == 1:
        raise ValueError(
            f'time size must = (1 / temp_res). (size: {time_size}, temp_res: {time_res}'
            )

    new_opt_func = partial(amp_f1_diff, DC_amp=c, time=time, f1_target=f1_target, freq=f)

    return new_opt_func


def find_real_f1(
        DC_amp: float, f1_target: float,
        time: Optional[Time[np.ndarray]] = None) -> opt.OptimizeResult:
    """Find pure sin amplitude for given empirical mean resp and empirical f1

    Done by minimisation.  If not successful, raises TypeError
    """

    time = Time(np.arange(1000), 'ms') if time is None else time

    obj_f = opt_f1(time=time, c=DC_amp, f1_target=f1_target)

    opt_results = minimize(obj_f, np.array([40]), method="Nelder-Mead")

    # check of optimisation success
    if not opt_results.success:
        raise TypeError('Real Unrectified amplitude could not be derived')

    return opt_results
