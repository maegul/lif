"""Utilities for inspecting or understanding aspects of this modelling project


"""

from functools import partial

import matplotlib.pyplot as plt
from numpy import fft
import numpy as np

from scipy.optimize import minimize

from ipywidgets import interact, fixed
# from ipywidgets import IntSlider, FloatSlider, interactive_output, HBox, VBox


# > sinusoid and fft


def gen_sin(a, c, t, f=10):
    """Generate sinusoid with t: 1->2pi

    Parameters
    ----
    a: amplitude
    c: translation
    t: time, in "seconds", with a frequency of 1 corresponding
        to a single cycle in 1 second of t
    f: frequency
    """

    return a * np.sin(f * 2 * np.pi * t) + c


def gen_fft(s, t, view_max=20):
    """Generate fft spectrum and frequency axis

    Parameters
    ----
    s: signal
    t: time dimension over which signal modulates
    view_max: max number of frequency increments returned

    Returns
    ----
    spectrum:
        normalised by size of signal array and mirroring of real signals
        in imaginary and complex (ie, multiplied by 2)
        Magnitude of 0Hz is not multiplied though, as not mirrored

        absolute values returned

    freq:
        frequencies corresponding to spectrum values
    """

    norm_factor = t.size

    spec = fft.rfft(s)
    # spec = fft.fftshift(fft.fft(s))
    spec[1:] *= 2 / norm_factor  # to normalise the split / coonjugate values
    spec[0] /= norm_factor
    spec = np.abs(spec)

    freq = fft.rfftfreq(t.size, (t[1] - t[0]))
    # freq = fft.fftfreq(t.size, (t[1]-t[0]))

    return spec[:view_max], freq[:view_max]


def gen_freq_report(n, d, print_rep=True):
    '''Reports description of freq range for a signal

    Should correspond with functions like `np.fft.rfftfreq`

    Parameters
    ----
    n : number of samples
    d : distance between samples (inverse of sample frequency)

    Returns
    ----
    min_freq, max_freq, num_freqs : float, float, int

    Notes
    ----
    min = 1 / (nd)
    If d = resolution and n = extent / resolution,
    then n * d = extent
    then min = 1 / extent

    step and min are the same

    max = 1 / (2d) if n is even
    max = (n-1) / (2dn) if n is odd

    n_freqs = floor(n / 2)
    '''

    min_freq = step_freq = 1 / (d * n)
    # depends on whether n is odd or even
    max_freq = 1 / (2 * d) if (n % 2 == 0) else (n - 1) / (2 * d * n)
    num_freqs = n // 2

    if print_rep:
        print(f"""
        min:  {min_freq:.3f}
        step: {step_freq:.3f}
        max:  {max_freq:.3f}
        num:  {num_freqs} (plus zero)
        """)

    return min_freq, max_freq, num_freqs


# >> Rectification and FFt

def show_rect_wave_fft(a, c, t, view_max, figsize=(15, 8)):
    """Graph effect of rectification on spectrum

    Parameters passed to gen_sin
    figsize -> plt.figure

    First graph: signal
    Second graph: spectra

    Signals:
        Full
        rectified at zero (rect)
        centered and rectified at zero (r_sub)
    """

    sin = gen_sin(a, c, t, f=10)

    sin_rect = sin.copy()
    sin_rect[sin_rect < 0] = 0

    sin_r_sub = sin - c
    sin_r_sub[sin_r_sub < 0] = 0

    spec, freq = gen_fft(sin, t, view_max=view_max)
    spec_rect, freq = gen_fft(sin_rect, t, view_max=view_max)

    spec_r_sub, freq = gen_fft(sin_r_sub, t, view_max=view_max)

    # mod ratio
    # 10 and 0, as with time window of 1000 and sample period of 1/1000, freq is in steps of 1
    sin_mr = spec[10] / spec[0]
    sin_rect_mr = spec_rect[10] / spec_rect[0]
    sin_r_sub_mr = spec_r_sub[10] / spec_r_sub[0]

    plt.figure(figsize=figsize)

    plt.subplot(121)
    plt.plot(sin, lw=2)
    plt.plot(sin_rect, ':', lw=3)
    plt.plot(sin_r_sub, '--', lw=3)

    plt.subplot(122)
    plt.plot(freq, spec)
    plt.plot(freq, spec_rect, label='rect')
    plt.plot(freq, spec_r_sub, label='r_sub')
    plt.legend()
    plt.title(f'mr: sin: {sin_mr:.2f}, r: {sin_rect_mr:.2f}, r_sub: {sin_r_sub_mr:.2f}')


# >>> Jupyter Dashboard
def gen_interactive_rect_fft(
        a=(0, 100, 1), c=(-5, 200, 0.1),
        view_max=(5, 100, 1),
        t=(0, 1, 1 / 1000),
        figsize=(15, 8)):
    """Generate interactive ipywidget dashboard of rectified signal and fft

    Argments passed to show_rect_wave_fft

    t tuple is passed to numpy as np.arange(*t).
    Thus t: (start, stop, step)

    t and figsize are passed to interact as fixed arguments

    """

    t = fixed(np.arange(*t))
    figsize = fixed(figsize)

    interact(show_rect_wave_fft, a=a, c=c, view_max=view_max, t=t, figsize=figsize)


# > Temporal Filter

# convolution viewer



# > F1 Calculation

def gen_rect_sig_fft(a, c, t, f=1, view_max=20):
    """generate rectified signal and spec for a(amp), c(trans) and t(time)"""

    signal = gen_sin(a, c, t, f=f)
    signal[signal < 0] = 0  # rectify signal

    spec, freq = gen_fft(signal, t, view_max=view_max)

    return signal, spec, freq


def gen_f1_diff(a, c, t, f1, f=1):
    """Find diff between target f1 and f1 of rectified sin

    a, c, t, f -> gen_sin

    f1: target f1

    frequency (f) is vital to this process, as it is the frequency that defines
    the F1 of the signal.
    frequency must be set so that it is guaranteed to be available from the fft.

    To do this, f must be set to an integer (1 here), AND,
    the total length of t (in time) must be 1, which ensures that the number of
    samples (n) and their period (d) are inversions of each other (n = 1/d).
    This ensures that the frequencies available from the fft are:
    0, 1, 2, ..., n/2-1, n/2 (see docs for np.fft.rfftfreq)
    """

    # signal = gen_sin(a, c, t, f=f)
    # signal[signal < 0] = 0  # rectify signal

    # spec, freq = gen_fft(signal, t)

    signal, spec, freq = gen_rect_sig_fft(a, c, t, f)

    assert np.isclose((t[1] - t[0]) * t.size, 1), (
        't (time) must have time length 1 and d = 1/n \n',
        f'Currently size (n): {t.size}, d: {t[1] - t[0]} and d*n: {t.size * (t[1]-t[0])}')

    assert isinstance(f, int), 'f must be an integer'

    # this works as the freqs are 0, 1, 2 (provided assertions above)
    # and, because the freqs start at 0, f can be used to index to find it's corresponding
    # position
    current_f1 = spec[f]

    # return abs difference betwee target (f1) and current f1
    return np.abs(current_f1 - f1)


# >> Optimisation function
def find_real_f1(c, f1, t=None, method="Nelder-Mead"):
    """Find amplitue of underlying sin for reported f1

    Uses mean full-field firing rate (c) and reported F1.
    Assuming rectification, the true underlying amplitude is determined
    using optimisation

    Returns
    ----
    result for a (amplitude) as a single value
    full optimisation results
    """

    t = (np.arange(0, 1, 1 / 1000)
         if t is None
         else t
         )

    assert np.isclose((t[1] - t[0]) * t.size, 1), (
        't (time) must have time length 1 and d = 1/n \n',
        f'Currently size (n): {t.size}, d: {t[1] - t[0]} and d*n: {t.size * (t[1]-t[0])}')

    # only free parameter is a (amplitude) ... as we're fitting a to given f1
    ojb_f = partial(gen_f1_diff, c=c, t=t, f1=f1, f=1)

    opt_results = minimize(ojb_f, np.array([40]), method=method)

    return opt_results.x[0], opt_results


# >> interactive

def show_f1_opt(a, c, t, f1, f, view_max=10, figsize=(12, 5)):

    # signal = gen_sin(a, c, t, f=f)
    # signal[signal < 0] = 0

    # spec, freq = gen_fft(signal, t, view_max=view_max)

    signal, spec, freq = gen_rect_sig_fft(a, c, t, f, view_max=view_max)
    current_f1 = spec[f]

    plt.figure(figsize=figsize)

    plt.subplot(121)
    plt.plot(t, signal)
    plt.axhline(c, c='0.65', ls=':')
    plt.ylim(top=a * 3)

    plt.subplot(122)
    plt.plot(freq, np.abs(spec))
    plt.axhline(f1, c='r', ls=':')

    plt.title(f'|{f1:.3f}-{current_f1:.3f}|={np.abs(f1-current_f1):.3f}')


def gen_interactive_f1_opt(
        f1, c, a=(0, 100, 0.5),
        view_max=10, f=1, t=(0, 1, 1 / 1000),
        figsize=(15, 8)):

    f1, c, f = fixed(f1), fixed(c), fixed(f)
    t = fixed(np.arange(*t))
    figsize = fixed(figsize)
    view_max = fixed(view_max)

    interact(
        show_f1_opt,
        a=a, c=c, t=t, f1=f1, f=f,
        view_max=view_max, figsize=figsize)
