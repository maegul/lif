"""Functions and classes for the generation of spatial and temporal filters
"""

from typing import Optional, Union, cast, Tuple

import numpy as np  # type: ignore

# import matplotlib.pyplot as plt

# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui

# import scipy.stats as st

from ...utils.units.units import SpatFrequency, TempFrequency, ArcLength, Time, val_gen
from . import data_objects as do

PI: float = np.pi
# cast(float, PI)


# > Spatial

def mk_spat_coords_1d(
        spat_res: ArcLength[float] = ArcLength(1, 'mnt'),
        spat_ext: ArcLength[float] = ArcLength(300, 'mnt')) -> ArcLength[np.ndarray]:
    """1D spatial coords symmetrical about 0 with 0 being the central discrete point

    Center point (value: 0) will be at index spat_ext // 2 if spat_res is 1
    Or ... size // 2
    """

    spat_radius = np.floor(spat_ext.mnt / 2)
    coords = ArcLength(
            np.arange(-spat_radius, spat_radius + spat_res.mnt, spat_res.mnt),
            'mnt'
        )

    return coords


def mk_sd_limited_spat_coords(
        sd: ArcLength[float],
        spat_res: ArcLength = ArcLength(1, 'mnt'),
        sd_limit: int = 5) -> ArcLength[np.ndarray]:
    """Wrap mk_spat_coords_1d to limit extent by number of SD

    SD is rounded to ceiling integer before multiplication
    """

    # integer rounded max sd times rf_sd_limit -> img limit for RF
    max_sd = sd_limit * np.ceil(sd.mnt)
    # spatial extent (for mk_coords)
    # add 1 to ensure number is odd so size of mk_coords output is as specified
    spat_ext = ArcLength((2 * max_sd) + 1, spat_res.unit)

    coords = mk_spat_coords_1d(spat_res=spat_res, spat_ext=spat_ext)

    return coords


def mk_blank_coords(
        spat_res: ArcLength = ArcLength(1, 'mnt'), temp_res: Time = Time(1, 'ms'),
        spat_ext: ArcLength = ArcLength(300, 'mnt'), temp_ext: Time = Time(1000, 'ms'),
        ) -> np.ndarray:
    '''
    Produces blank coords for each spatial and temporal value
    that would be ordinarily created
    Base units are minutes of arc and milliSeconds

    Parameters
    ----
    spat_res : float
        resolution, in minutes of arc, of meshgrid
    temp_res : float
        resolution, in milliseconds, of temporal dimension of meshgrid
    spat_ext : int
        Width and height, in minutes of arc, of spatial dimensions of meshgrid
        spat_ext is the total width (horiztontal or vertical) of the stimulus
        image generated.
        radial extent (horizontal or vertical, from center) will be
        floor(spat_ext/2), and always centred on zero.
        So actual extent will be 1*spat_res greater for even and as specified
        for odd extents
    temp_ext : int
        duration, in milliseconds, of temporal dimension

    Returns
    ----
    single meshgrid (3D: x,y,t)
    '''

    # spat_radius = np.floor(spat_ext / 2)
    # x_coords = np.arange(-spat_radius, spat_radius + spat_res, spat_res)

    x_coords = mk_spat_coords_1d(spat_res, spat_ext)
    # treat as image (with origin at top left or upper)
    # y_cords positive at top and negative at bottom
    y_coords = ArcLength(x_coords.mnt[::-1], 'mnt')

    t_coords = Time(np.arange(0, temp_ext.ms, temp_res.ms), 'ms')

    # ie, one array with appropriate size, each coordinate represented by a single value
    space: np.ndarray
    space = np.zeros((y_coords.mnt.size, x_coords.mnt.size, t_coords.ms.size))  # type: ignore

    return space


def mk_spat_coords(
        spat_res: ArcLength = ArcLength(1, 'mnt'),
        spat_ext: ArcLength = ArcLength(300, 'mnt'),
        ) -> Tuple[ArcLength[np.ndarray], ArcLength[np.ndarray]]:

    '''
    Produces spatial (ie meshgrid) for
    generating RFs and stimuli
    Base units are minutes of arc and milliSeconds

    Parameters
    ----
    spat_res : float
        resolution, in minutes of arc, of meshgrid
    spat_ext : int
        Width and height, in minutes of arc, of spatial dimensions of meshgrid
        spat_ext is the total width (horiztontal or vertical) of the stimulus
        image generated.
        radial extent (horizontal or vertical, from center) will be
        floor(spat_ext/2), and always centred on zero.
        So actual extent will be 1*spat_res greater for even and as specified
        for odd extents

    Returns
    ----
    xc, yc

    all meshgrids filled with appropriate coordinate values
    '''

    # spat_radius = np.floor(spat_ext / 2)
    # x_coords = np.arange(-spat_radius, spat_radius + spat_res, spat_res)

    x_coords = mk_spat_coords_1d(spat_res, spat_ext)
    # treat as image (with origin at top left or upper)
    # y_cords positive at top and negative at bottom
    y_coords = ArcLength(x_coords.mnt[::-1], 'mnt')

    xc: ArcLength[np.ndarray]
    yc: ArcLength[np.ndarray]
    xc, yc = (
        ArcLength(c, 'mnt') for c in np.meshgrid(x_coords.mnt, y_coords.mnt)  # type: ignore
        )

    return xc, yc


def mk_spat_temp_coords(
        spat_res: ArcLength = ArcLength(1, 'mnt'), temp_res: Time = Time(1, 'ms'),
        spat_ext: ArcLength = ArcLength(300, 'mnt'), temp_ext: Time = Time(1000, 'ms'),
        ) -> Tuple[ArcLength[np.ndarray], ArcLength[np.ndarray], Time[np.ndarray]]:
    '''
    Produces spatial and temporal coordinates (ie meshgrid) for
    generating RFs and stimuli
    Base units are minutes of arc and milliSeconds

    Parameters
    ----
    spat_res : float
        resolution, in minutes of arc, of meshgrid
    temp_res : float
        resolution, in milliseconds, of temporal dimension of meshgrid
    spat_ext : int
        Width and height, in minutes of arc, of spatial dimensions of meshgrid
        spat_ext is the total width (horiztontal or vertical) of the stimulus
        image generated.
        radial extent (horizontal or vertical, from center) will be
        floor(spat_ext/2), and always centred on zero.
        So actual extent will be 1*spat_res greater for even and as specified
        for odd extents
    temp_ext : int
        duration, in milliseconds, of temporal dimension

    Returns
    ----
    xc, yc, tc

    all meshgrids filled with appropriate coordinate values
    '''

    # spat_radius = np.floor(spat_ext / 2)
    # x_coords = np.arange(-spat_radius, spat_radius + spat_res, spat_res)

    x_coords = mk_spat_coords_1d(spat_res, spat_ext)
    # treat as image (with origin at top left or upper)
    # y_cords positive at top and negative at bottom
    y_coords = ArcLength(x_coords.mnt[::-1], 'mnt')

    t_coords = Time(np.arange(0, temp_ext.ms, temp_res.ms), 'ms')

    _xc: np.ndarray
    _yc: np.ndarray
    _tc: np.ndarray

    _xc, _yc, _tc = np.meshgrid(x_coords.mnt, y_coords.mnt, t_coords.ms)  # type: ignore
    xc, yc = (ArcLength(c, 'mnt') for c in (_xc, _yc))
    tc = Time(_tc, 'ms')

    return xc, yc, tc


def mk_gauss_1d(
        coords: ArcLength[np.ndarray],
        mag: float = 1, sd: ArcLength[float] = ArcLength(10, 'mnt')) -> np.ndarray:
    "Simple 1d gauss ... should be possible to multiply with another for 2d"

    gauss_coeff = mag / (sd.mnt * (2*PI)**0.5)  # ensure sum of 1
    # gauss_1d: np.ndarray  # as coords SHOULD always be an np.array
    gauss_1d = gauss_coeff * np.exp(-coords.mnt**2 / (2 * sd.mnt**2))  # type: ignore

    return gauss_1d


def mk_gauss_1d_ft(
        freqs: SpatFrequency[np.ndarray],
        amplitude: float = 1, sd: ArcLength[float] = ArcLength(10, 'mnt')) -> np.ndarray:
    """Returns normalised ft, treats freqs as  cycs per minute

    Works with mk_gauss_1d

    Presumes normalised gaussian (relies on such for mk_gauss_1d).
    Thus coefficient is 1

    freqs: cpm

    Question of what amplitude is ... arbitrary amplitude of spatial filter??
    """
    ft = amplitude * np.exp(-PI**2 * freqs.cpm**2 * 2 * sd.mnt**2)

    # Note: FT = T * FFT ~ amplitude of convolution
    return ft


def mk_gauss_2d(
        x_coords: ArcLength, y_coords: ArcLength,
        gauss_params: do.Gauss2DSpatFiltParams) -> np.ndarray:

    gauss_x = mk_gauss_1d(x_coords, sd=gauss_params.arguments.h_sd)
    gauss_y = mk_gauss_1d(y_coords, sd=gauss_params.arguments.v_sd)

    gauss_2d = gauss_params.amplitude * gauss_x * gauss_y

    return gauss_2d


def mk_gauss_2d_ft(
        freqs_x: SpatFrequency, freqs_y: SpatFrequency,
        gauss_params: do.Gauss2DSpatFiltParams) -> np.ndarray:

    # amplitude is 1 for 1d so that amplitude for 2d applies to whole 2d
    gauss_ft_x = mk_gauss_1d_ft(freqs_x, sd=gauss_params.arguments.h_sd, amplitude=1)
    gauss_ft_y = mk_gauss_1d_ft(freqs_y, sd=gauss_params.arguments.v_sd, amplitude=1)

    gauss_2d_ft = gauss_params.amplitude * gauss_ft_x * gauss_ft_y

    # Note: FT = T * FFT ~ amplitude of convolution
    return gauss_2d_ft


# sf = spatial filter
# abbreviated as not intended for public facing use
def mk_dog_sf(
        x_coords: ArcLength[np.ndarray], y_coords: ArcLength[np.ndarray],
        dog_args: do.DOGSpatFiltArgs) -> np.ndarray:

    cent_gauss_2d = mk_gauss_2d(x_coords, y_coords, gauss_params=dog_args.cent)
    surr_gauss_2d = mk_gauss_2d(x_coords, y_coords, gauss_params=dog_args.surr)

    return cent_gauss_2d - surr_gauss_2d


def mk_dog_sf_ft(
        freqs_x: SpatFrequency, freqs_y: SpatFrequency,
        dog_args: do.DOGSpatFiltArgs) -> np.ndarray:

    cent_ft = mk_gauss_2d_ft(freqs_x, freqs_y, gauss_params=dog_args.cent)
    surr_ft = mk_gauss_2d_ft(freqs_x, freqs_y, gauss_params=dog_args.surr)

    dog_rf_ft = cent_ft - surr_ft

    # Note: FT = T * FFT ~ amplitude of convolution
    return dog_rf_ft


def mk_dog_sf_ft_1d(
        freqs: SpatFrequency,
        dog_args: Union[do.DOGSpatFiltArgs, do.DOGSpatFiltArgs1D]) -> np.ndarray:
    """Make 1D Fourier Transform of DoG RF but presume radial symmetry and return only 1D

    dog_args: if full 2d DOGSpatFiltArgs then uses DOGSpatFiltArgs.to_dog_1d() to make 1d
    """

    if isinstance(dog_args, do.DOGSpatFiltArgs):
        dog_args_1d: do.DOGSpatFiltArgs1D = dog_args.to_dog_1d()
    else:
        dog_args_1d: do.DOGSpatFiltArgs1D = dog_args

    cent_ft = mk_gauss_1d_ft(freqs, amplitude=dog_args_1d.cent.amplitude, sd=dog_args_1d.cent.sd)
    surr_ft = mk_gauss_1d_ft(freqs, amplitude=dog_args_1d.surr.amplitude, sd=dog_args_1d.surr.sd)

    dog_ft_1d = cent_ft - surr_ft

    return dog_ft_1d


# > Temporal

def mk_temp_coords(
        temp_res: Time[float], temp_ext: Time[float],
        temp_ext_n_tau: Optional[float] = None,
        tau: Optional[Time[float]] = None) -> Time[np.ndarray]:
    """Generate array of times with resolution temp_res and extent temp_ext

    Temporal parameters must be Time objects.
    Calculations all done in milliseconds (Time.ms)

    temp_ext_n_tau and tau are optional parameters for
    programmatically controlling the extent.
    If both provided, temp_ext = tau.ms * temp_ext_n_tau (in ms units)
    """

    if temp_ext_n_tau and tau:
        assert temp_ext_n_tau >= 10, (
            f'temp_ext_n_tau ({temp_ext_n_tau}) should be 10 or more\n'
            f'Any lower and the sum of the filter will be diminished \n'
            'by approx 0.1% or more for tq filt')

        temp_ext = Time(tau.ms * temp_ext_n_tau, 'ms')

    # converting to seconds from milliseconds
    # parameters of teich and qian function are in seconds
    t = Time(np.arange(0, temp_ext.ms, temp_res.ms), 'ms')

    return t


# >> tq temp filt

# this is a hidden function because temp filters are simpler than spatial
# thus, spatial don't need hidden as simpler 1D functions are used for fitting
# for temporal, to suit the array args needed for fitting, a separation
# of all the args is helpful

def _mk_tqtempfilt(
        t: Time[val_gen],
        tau: Time[float] = Time(16, 'ms'),
        w: float = 4 * 2 * np.pi,
        phi: float = 0.24) -> val_gen:
    r"""Generate single temp filter by modulating a negative exp with a cosine

    Parameters
    ----
    tau : float (milliseconds)
        Time constant of negative exponential
    w : float (n*2*pi radians)
        Frequency of modulation of negative exp
        As in "n*2*pi" units, so that frequency is in Hertz
    phi : float (0.24)
        Phase translation of modulation

    Returns
    ----
    tf : (temporal filter) (array 1D)
        Magnitudes of the filter over time for each time step defined by
        the parameters

    Notes
    ----
    Taken from Teich and Qian (2006) and Chen et al (2001)

    On time constant and decay/growth dynamics for this filter, looking only at the
    gamma function or exponential component, the integral is:

    $$F(t) = \int \frac{t \cdot exp(\frac{-t}{\tau})}{\tau^2} dt =
    -\frac{exp(\frac{-t}{\tau})(\tau + t)}{\tau}$$

    The definite integral from zero to a number of time constants \(n\tau\) is:

    $$ \begin{aligned}
        F(t)|^{n\tau}_{0} &= -\frac{\tau+n\tau}{\tau}exp(\frac{-n\tau}{\tau})
        - (-)\frac{\tau}{\tau}\exp(\frac{0}{\tau}) \\
                                            &= -(1+n)exp(-n) + 1 \\
                                            &= 1 - \frac{1+n}{e^n} \\
                                            &= 1 - \frac{1}{e^n} - \frac{n}{e^n}
    \end{aligned} $$

    Here, \(1 - \frac{1}{e^n}\) is ordinary exponential growth/decay.
    The additional term of \(- \frac{n}{e^n}\) slows the growth/decay, but converges
    to zero by approx. 10 time constants (0.05%)
    """

    # if temp_ext_n_tau:

    #     assert temp_ext_n_tau >= 10, (
    #         f'temp_ext_n_tau ({temp_ext_n_tau}) should be 10 or more\n'
    #         f'Any lower and the sum of the filter will be diminished by approx 0.1% or more')
    #     temp_ext = tau * temp_ext_n_tau

    # converting to seconds from milliseconds
    # parameters of teich and qian function are in seconds
    # t = np.arange(0, temp_ext, temp_res) / 1000
    # tau /= 1000

    # Note correction from teich and qian (apparent in Kuhlman as well as Chen(?) too)
    exp_term: val_gen = np.exp(-t.s / tau.s)  # type: ignore
    cos_term: val_gen = np.cos((w * t.s) + phi)  # type: ignore

    tf = (t.s / tau.s**2) * exp_term * cos_term

    return tf


def mk_tq_tf(
        t: Time[val_gen],
        tf_params: do.TQTempFiltParams) -> val_gen:
    r"""Generate single temp filter by modulating a negative exp with a cosine

    Args passed to _mk_tqtempfilt

    Parameters
    ----
    t: time
    tf_params: data object with amp, tau, w, phi


    Returns
    ----
    tf : (temporal filter) (array 1D)
        Magnitudes of the filter over time for each time step defined by
        the parameters

    Notes
    ----
    Taken from Teich and Qian (2006) and Chen et al (2001)

    On time constant and decay/growth dynamics for this filter, looking only at the
    gamma function or exponential component, the integral is:

    $$F(t) = \int \frac{t \cdot exp(\frac{-t}{\tau})}{\tau^2} dt =
    -\frac{exp(\frac{-t}{\tau})(\tau + t)}{\tau}$$

    The definite integral from zero to a number of time constants \(n\tau\) is:

    $$ \begin{aligned}
        F(t)|^{n\tau}_{0} &= -\frac{\tau+n\tau}{\tau}exp(\frac{-n\tau}{\tau})
        - (-)\frac{\tau}{\tau}\exp(\frac{0}{\tau}) \\
                                            &= -(1+n)exp(-n) + 1 \\
                                            &= 1 - \frac{1+n}{e^n} \\
                                            &= 1 - \frac{1}{e^n} - \frac{n}{e^n}
    \end{aligned} $$

    Here, \(1 - \frac{1}{e^n}\) is ordinary exponential growth/decay.
    The additional term of \(- \frac{n}{e^n}\) slows the growth/decay, but converges
    to zero by approx. 10 time constants (0.05%)
    """

    tf = tf_params.amplitude * _mk_tqtempfilt(
                    t,
                    tau=tf_params.arguments.tau,
                    w=tf_params.arguments.w,
                    phi=tf_params.arguments.phi
                )

    return tf


def _mk_tqtempfilt_ft(
        f: TempFrequency[val_gen],
        tau: Time[float] = Time(16, 'ms'),
        w: float = 4 * 2 * np.pi,
        phi: float = 0.24,
        return_abs: bool = True) -> val_gen:
    """Cerate Fourier Transform of function used in mk_tqtempfilt

    Employs analytical solution from Chen (2001)

    Parameters
    ----
    f : TempFrequency
        Frequencies to calculate magnitude of in Fourier Transform
        Used in cycs per radian
    tau, w, phi : as in mk_tqtempfilt
        tau is transformed to seconds but taken in milliseconds
    return_abs : boolean (True)
        Whether to return the absolute of the complex fourier
        transform array.
        This represents the magnitude of the fourier, and so is
        True be default

    Returns
    ----
    fourier : array

    Notes
    ----
    For the magnitude of this fourier transform to correspond to that of a DFT (such as that
    computed by a FFT) you must divide it by the temporal period or sampling interval of the
    original filter or signal (ie, the tq filt being used in calculating the DFT)

    $$DFT(FFT) \\Leftrightarrow \\frac{FT_{continuous}}{T}$$

    This magnitude will also correspond with the result of convolving a sinusoid of the
    same frequency with the same original filter or signal.  IE, the magnitude of that
    frequency after filtering by this filter.

    Does not take a discrete data object as used in fitting and optmisation, which
    requires taking an array of values.
    """

    # tau /= 1000

    a: val_gen
    b: val_gen

    a = np.exp(phi * 1j) / (1 / tau.s + ((f.w - w) * 1j))**2  # type: ignore
    b = np.exp(-phi * 1j) / (1 / tau.s + ((f.w + w) * 1j))**2  # type: ignore

    fourier: val_gen
    fourier = (1 / (2 * tau.s**2)) * (a + b)  # type: ignore
    if return_abs:
        fourier: val_gen = np.abs(fourier)  # type: ignore

    return fourier


def mk_tq_tf_ft(
        freqs: TempFrequency[val_gen],
        tf_params: do.TQTempFiltParams) -> val_gen:
    """Generates fourer of temporal filter generated by mk_tq_tf

    Arguments passed to _mk_tqtempfilt_ft
    Fourer is analytical

    Parameters
    ----
    freqs: frequencies to generate fourier for
    tf_params: parameters defined by data object

    Returns
    ----
    fourier of tq temp filt
    """

    tf_ft = tf_params.amplitude * _mk_tqtempfilt_ft(
                                        f=freqs,
                                        tau=tf_params.arguments.tau,
                                        w=tf_params.arguments.w,
                                        phi=tf_params.arguments.phi
                                    )

    return tf_ft


# >> Old Gauss (worgoter & Koch)

def mk_tempfilt(tau=10, temp_ext=100, temp_res=1, temp_ext_n_tau=None, return_t=False):
    '''Generate a temporal filter

    All time parameters in milliseconds

    Parameters
    ----
    tau : float
        Time constant
    temp_ext : float
        Length of temporal filter
    temp_res : float
        resolution of temporal filter
        size of temporal increment in milliseconds
    temp_ext_n_tau : int (Default None)
        length of temporl filter but in number of tau (time constants)
        If None, temp_ext used instead, if int, this is used
    return_t : Boolean
        If True, return time units array as well as magnitude of filer
        Else, return just the filter

    Maths and values taken from worgotter and koch (1991)
    '''

    # Use number of time consts as measure of temp extend, if specified
    if temp_ext_n_tau:
        temp_ext = temp_ext_n_tau * tau

    t = np.arange(0, temp_ext, temp_res)
    tf = np.exp(-t / tau) / tau

    if return_t:
        return t, tf
    else:
        return tf


def mk_tempfilt2(
        tau1=10, tau2=20,
        temp_res=1, temp_ext_n_tau=5, return_t=True,
        correct_integral_errors=True):
    """Generate two temporal filters with different time constants

    As with mk_tempfilt but with two taus


    Parameters
    ----
    correct_integral_errors : Boolean
        Ensure that both temporal filters sum to 1
        Necessary due to discretisation errors and need
        to ensure both filters do not alter the magnitude of the
        input signal.
        Done by dividing the temporal filter by the sum of the filter

    Returns
    ----
    tf1, tf2 : array (1D)
        EAch as the output of mk_tempfilt
        Magnitude of the filter at each time step in 1D filter
    """

    temp_ext = max([tau1, tau2]) * temp_ext_n_tau
    t, tf1 = mk_tempfilt(tau1, temp_ext, temp_res=temp_res, return_t=True)
    tf2 = mk_tempfilt(tau2, temp_ext, temp_res=temp_res)  # type: ignore
    tf1: np.ndarray
    tf2: np.ndarray

    if correct_integral_errors:
        tf1 /= tf1.sum()
        tf2 /= tf2.sum()

    if return_t:
        return t, tf1, tf2
    else:
        return tf1, tf2

