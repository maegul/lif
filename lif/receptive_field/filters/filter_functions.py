"""Functions and classes for the generation of spatial and temporal filters
"""

from typing import Union, cast, Tuple

import numpy as np  # type: ignore

# import matplotlib.pyplot as plt

# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui

import scipy.stats as st

from ...utils.units.units import SpatFrequency, TempFrequency
from . import data_objects as do

PI: float = np.pi
# cast(float, PI)

def mk_spat_coords_1d(spat_res: float = 1, spat_ext: float = 300) -> np.ndarray:
    """1D spatial coords symmetrical about 0 with 0 being the central discrete point

    Center point (value: 0) will be at index spat_ext // 2 if spat_res is 1
    Or ... size // 2
    """

    spat_radius = np.floor(spat_ext / 2)
    coords = np.arange(-spat_radius, spat_radius + spat_res, spat_res)

    return coords


def mk_sd_limited_spat_coords(
        sd: float, spat_res: float = 1, sd_limit: int = 5):

    # integer rounded max sd times rf_sd_limit -> img limit for RF
    max_sd = sd_limit * np.ceil(sd)
    # spatial extent (for mk_coords)
    # add 1 to ensure number is odd so size of mk_coords output is as specified
    spat_ext = (2 * max_sd) + 1

    coords = mk_spat_coords_1d(spat_res=spat_res, spat_ext=spat_ext)

    return coords


def mk_coords(
        spat_res=1, temp_res=1,
        spat_ext=300, temp_ext=1000,
        temp_dim=True, blank_grid=False
        ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
    temp_dim : boolean
        Whether to include a temporal dimension in the meshgrid
    blank_grid : boolean
        Whether to return a meshhgrid with all values being zero
        Useful for filling in the grid manually for stimulus creation

    Returns
    ----
    xc, yc, tc (if temp_dim)
    xc, yc if temp_dim==false

    all meshgrids filled with appropriate coordinate values
    If blank_grid, then single meshgrid (3D: x,y,t)
    '''

    # spat_radius = np.floor(spat_ext / 2)
    # x_coords = np.arange(-spat_radius, spat_radius + spat_res, spat_res)

    x_coords = mk_spat_coords_1d(spat_res, spat_ext)
    # treat as image (with origin at top left or upper)
    # y_cords positive at top and negative at bottom
    y_coords = x_coords[::-1]

    t_coords = np.arange(0, temp_ext, temp_res)

    # ie, one array with appropriate size, each coordinate represented by a single value
    if blank_grid:
        space = np.zeros((y_coords.size, x_coords.size, t_coords.size))
        return space

    if not temp_dim:
        xc, yc = np.meshgrid(x_coords, y_coords)
        return xc, yc
    else:
        xc, yc, tc = np.meshgrid(x_coords, y_coords, t_coords)
        return xc, yc, tc



def mk_gauss_1d(
        coords: np.ndarray,
        mag: float = 1, sd: float = 10) -> np.ndarray:
    "Simple 1d gauss ... should be possible to multiply with another for 2d"

    gauss_coeff = mag / (sd * (2*PI)**0.5)  # ensure sum of 1
    gauss_1d = gauss_coeff * np.exp(-coords**2 / (2 * sd**2))

    return gauss_1d


def mk_gauss_1d_ft(
        freqs: SpatFrequency,
        amplitude: float = 1, sd: float = 10) -> np.ndarray:
    """Returns normalised ft, treats freqs as being in hz

    Works with mk_gauss_1d

    Presumes normalised gaussian (relies on such for mk_gauss_1d).
    Thus coefficient is 1

    freqs: cpm

    Question of what amplitude is ... arbitrary amplitude of spatial filter??
    """
    ft = amplitude * np.exp(-PI**2 * freqs.cpm**2 * 2 * sd**2)  # type: ignore

    # Note: FT = T * FFT ~ amplitude of convolution
    return ft


def mk_gauss_2d(
        x_coords: np.ndarray, y_coords: np.ndarray,
        gauss_params: do.Gauss2DSpatFiltParams) -> np.ndarray:

    amplitude, h_sd, v_sd = gauss_params.array()

    gauss_x = mk_gauss_1d(x_coords, sd=h_sd)
    gauss_y = mk_gauss_1d(y_coords, sd=v_sd)

    gauss_2d = amplitude * gauss_x * gauss_y

    return gauss_2d


def mk_gauss_2d_ft(
        freqs_x: TempFrequency, freqs_y: TempFrequency,
        gauss_params: do.Gauss2DSpatFiltParams):

    amplitude, h_sd, v_sd = gauss_params.array()

    # amplitude is 1 for 1d so that amplitude for 2d applies to whole 2d
    gauss_ft_x = mk_gauss_1d_ft(freqs_x, sd=h_sd, amplitude=1)
    gauss_ft_y = mk_gauss_1d_ft(freqs_y, sd=v_sd, amplitude=1)

    gauss_2d_ft = amplitude * gauss_ft_x * gauss_ft_y

    # Note: FT = T * FFT ~ amplitude of convolution
    return gauss_2d_ft


def mk_dog_rf(
        x_coords: np.ndarray, y_coords: np.ndarray,
        dog_args: do.DOGSpatFiltArgs
        ):

    cent_gauss_2d = mk_gauss_2d(x_coords, y_coords, gauss_params=dog_args.cent)
    surr_gauss_2d = mk_gauss_2d(x_coords, y_coords, gauss_params=dog_args.surr)

    return cent_gauss_2d - surr_gauss_2d


def mk_dog_rf_ft(
        freqs_x: TempFrequency, freqs_y: TempFrequency,
        dog_args: do.DOGSpatFiltArgs):

    cent_ft = mk_gauss_2d_ft(freqs_x, freqs_y, gauss_params=dog_args.cent)
    surr_ft = mk_gauss_2d_ft(freqs_x, freqs_y, gauss_params=dog_args.surr)

    dog_rf_ft = cent_ft - surr_ft

    # Note: FT = T * FFT ~ amplitude of convolution
    return dog_rf_ft


def mk_dog_rf_ft_1d(
        freqs: TempFrequency,
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


def mk_rf(
        spat_res: float = 1,
        cent_h_sd: float = 10.6, cent_v_sd: float = 10.6,
        surr_h_sd: float = 31.8, surr_v_sd: float = 31.8,
        mag_cent: float = 1, mag_surr: float = 1,
        rf_sd_limit: int = 5,
        return_cent_surr: bool = False, return_coords: bool = False):
    '''Generate DoG Rec Field

    Parameters
    ----
    spat_res : float
        resolution of meshgrid used to create RF
        passed to mk_coords
    cent/surr_h/v_sd : float
        standard dev of gaussian used to generate RF
        cent/surr -> center or surround component
        h/v : vertical or horizontal axis
    mag_cent/surr : float
        absolute magnitude of cent/surr component
    rf_sd_limit : float
        span of underlying mesdhgrid in number of standard deviations
        the maximum of of all cent/surr_h/v_sd are used
        important to ensure that the sum of the surround is not arbitrarily
        trunacated
    return_cent_surr : boolean
        whether to return the center and surround components also
    return_coords : boolean
        whether to also return the meshgrid for x and y

    Units are defined in arc minutes

    Current defaults from Woergoetter and Koch (1991)

    Returns
    ----

    '''

    # integer rounded max sd times rf_sd_limit -> img limit for RF
    max_sd = rf_sd_limit * np.ceil(
        np.max([cent_h_sd, cent_v_sd, surr_h_sd, surr_v_sd])
    )
    # spatial extent (for mk_coords)
    # add 1 to ensure number is odd so size of mk_coords output is as specified
    spat_ext = (2 * max_sd) + 1

    # only one coords necessary, as square/cartesion grid (how to noise hexagonal centering?)
    x_coords, y_coords = mk_coords(spat_res=spat_res, spat_ext=spat_ext, temp_dim=False)

    rf_cent = (
        # mag divide by normalising factor with both sds (equivalent to sq if they were identical)
        (mag_cent / (2 * np.pi * cent_v_sd * cent_h_sd)) *
        np.exp(
            - (
                (x_coords**2 / (2 * cent_h_sd**2)) +
                (y_coords**2 / (2 * cent_v_sd**2))
            )
        )
    )

    rf_surr = (
        (mag_surr / (2 * np.pi * surr_v_sd * surr_h_sd)) *
        np.exp(
            - (
                (x_coords**2 / (2 * surr_h_sd**2)) +
                (y_coords**2 / (2 * surr_v_sd**2))
            )
        )
    )

    rf = rf_cent - rf_surr

    if not (return_cent_surr or return_coords):

        return rf

    else:
        ret_vars = (rf,)

        if return_cent_surr:
            ret_vars += rf_cent, rf_surr
        if return_coords:
            ret_vars += x_coords, y_coords

        return ret_vars


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
    tf2 = mk_tempfilt(tau2, temp_ext, temp_res=temp_res)

    if correct_integral_errors:
        tf1 /= tf1.sum()
        tf2 /= tf2.sum()

    if return_t:
        return t, tf1, tf2
    else:
        return tf1, tf2


def mk_tqtempfilt(
        tau=16, w=4 * 2 * np.pi, phi=0.24,
        temp_ext=100, temp_ext_n_tau=None,
        temp_res=1, return_t=True):
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
    temp_ext : int
        Number of temp_res units to evaluate filter for.
        time array will be np.arange(0, temp_ext, temp_res)
    temp_ext_n_tau : int (default None)
        Substitute temp_ext with a multiple of time constant (tau)
        temp_ext = temp_ext_n_tau * tau
        Approx >=10 time constants of evaluation is appropriate.
        Below 10, the integral or sum diminishes nonnegligibly.
        Time constant dynamics are similar to ordinary negative exponential
        but slower: see Notes.
    temp_res : float (milliseconds) (default 1)
        Temporal resolution of the filter
    return_t : boolean
        Whether to return the time array along with the filter array

    Returns
    ----
    tf : (temporal filter) (array 1D)
        Magnitudes of the filter over time for each time step defined by
        the parameters
    t : (time) (array 1D)
        Time steps for which filter is calculated
        Only if `return_t=True`


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

    if temp_ext_n_tau:

        assert temp_ext_n_tau >= 10, (
            f'temp_ext_n_tau ({temp_ext_n_tau}) should be 10 or more\n'
            f'Any lower and the sum of the filter will be diminished by approx 0.1% or more')
        temp_ext = tau * temp_ext_n_tau

    # converting to seconds from milliseconds
    # parameters of teich and qian function are in seconds
    t = np.arange(0, temp_ext, temp_res) / 1000
    tau /= 1000

    # Note correction from teich and qian (apparent in Kuhlman as well as Chen(?) too)
    tf = (t / tau**2) * np.exp(-t / tau) * np.cos((w * t) + phi)

    if return_t:
        return tf, t
    else:
        return tf


def mk_tq_ft(
        f: Union[float, np.ndarray], 
        tau: float = 16, 
        w: float = 4 * 2 * np.pi, 
        phi: float = 0.24, 
        return_abs: bool = True) -> np.ndarray:
    """Cerate Fourier Transform of function used in mk_tqtempfilt

    Employs analytical solution from Chen (2001)

    Parameters
    ----
    f : float/array
        Frequencies to calculate magnitude of in Fourier Transform
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
    """

    tau /= 1000

    a = np.exp(phi * 1j) / (1 / tau + ((f - w) * 1j))**2
    b = np.exp(-phi * 1j) / (1 / tau + ((f + w) * 1j))**2

    fourier = (1 / (2 * tau**2)) * (a + b)
    if return_abs:
        fourier = np.abs(fourier)

    return fourier
