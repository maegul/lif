"""Module for generation of stimuli


"""

import numpy as np
import matplotlib.pyplot as plt

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui


def mk_coords(
        spat_res=1, temp_res=1,
        spat_ext=300, temp_ext=1000,
        temp_dim=True, blank_grid=False):
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

    spat_radius = np.floor(spat_ext / 2)
    x_coords = np.arange(-spat_radius, spat_radius + spat_res, spat_res)
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


def mk_rf(
        spat_res=1,
        cent_h_sd=10.6, cent_v_sd=10.6, surr_h_sd=31.8, surr_v_sd=31.8,
        mag_cent=1, mag_surr=1,
        rf_sd_limit=5,
        return_cent_surr=False, return_coords=False):
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


def mk_tq_fft(f, tau=16, w=4 * 2 * np.pi, phi=0.24, return_abs=True):
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


def mk_sinstim(
        ori=90, spat_freq=1, temp_freq=1,
        spat_ext=120, temp_ext=1000,
        spat_res=1, temp_res=1):
    '''Generate sinusoidal grating

    Parameters
    ----

    ori : float (degrees)
    spat_freq : float
        spatial frequency of the sinusoidal grating in cycles/degree
    temp_freq : float
        temporal frequency in cycles per second
        how many cycles are traversed in one second of time
        how many cycles actually traversed in output depends on temp_ext also
    spat_ext, temp_ext, spat_res, temp_res : float
        passed to mk_coords


    Unit conversions, from base of min of arc and milliSecs to requisite degrees-> radians
    and milliseconds -> seconds is done within this function.
    BIG PRESUMPTION here (important!) is that all units coming from the mk_coords func
    are in minutes and milliSecs and so need to be converted for trigonometric functions
    used here.

    '''

    # Orientation to radians
    # add 90 so 0degs -> 3-o'clock, 90degs -> 12-o'clock
    # depends on how scaling is done below
    ori = np.radians(ori + 90)

    # Spatial Frequency
    # multiply by 2pi to change to whole cycles per unit
    k = spat_freq * 2 * np.pi  # 1 cpd (?)

    # X & Y scaling factors to incorporate orientation
    # scale x coords by cos so constant at 90 degs (at which cos=0)
    # scale y coords by sin so constant at 0 degs (at chich sin=0)
    # then, as ori += 90, image rotated
    # Movement through time (tc) is in +90degs direction,
    # as coords are centered on zero and negative in bottom and left
    # and tc coords are subtracted below
    kx = k * np.cos(ori)
    ky = k * np.sin(ori)

    # Temporal Frequency
    # again, multiply by 2pi to be cycles per unit time
    omega = temp_freq * 2 * np.pi

    # creating coords
    xc, yc, tc = mk_coords(
        spat_ext=spat_ext, temp_ext=temp_ext,
        spat_res=spat_res, temp_res=temp_res
    )

    # Convert spatial coords to degs (from minutes)
    # Convert temporal coords to secs (from milliSecs)
    # scale coords so that unit -> spat_freq, temp_freq cycles
    xc = kx * (xc / 60)
    yc = ky * (yc / 60)
    tc = omega * tc / 1000

    img = 0.5 + 0.5 * np.cos(xc + yc - tc)

    return img


def mk_barstim(
        height=None, width=None, speed=None, sweep=None,
        mean_lum=0.5, bar_lum=1, max_contrast=False,
        spat_ext=120, temp_ext=1000,
        spat_res=1, temp_res=1):
    '''
    Spatial and temporal units are all in minutes and milliSecs

    speed must be in minutes / milliSec (convert from deg/S by *60/1000)

    mean_lum : float (0-1)
        background value of returned image ... the mean luminance
        of the stimulus.
        Must be between zero (0) and one (1)
    bar_lum : float (0-1)
        value of the bar stimulus between 0 and 1
    max_contrast : boolean
        whether to always use maximum contrast for the background
        and ignore mean_lum.  ie, if bar_lum is 1, background is 0
        and vice versa.
    spat_ext, temp_ext, spat_res, temp_res : float
        passed to mk_coords
    '''

    # creating stim_space (y,x,t)
    stim_space = mk_coords(
        blank_grid=True,
        spat_ext=spat_ext, temp_ext=temp_ext,
        spat_res=spat_res, temp_res=temp_res
    )

    if max_contrast:
        # pick whichever of (0,1) is furtherest away from bar lum
        mean_lum = 1 if bar_lum < 0.5 else 0

    stim_space += mean_lum

    # Round, convert to minutes and ensure odd so that always has a center
    height = round(height)
    if height % 2 == 0:
        height += 1

    width = round(width)
    if width % 2 == 0:
        width += 1

    sweep = round(sweep)
    if sweep % 2 == 0:
        sweep += 1

    img_rad = (stim_space.shape[0] - 1) / 2  # also idx of cent
    h_rad = (height - 1) / 2  # also idx of cent
    w_rad = int((width - 1) / 2)
    sweep_rad = (sweep - 1) / 2

    h_bounds = np.array([img_rad - h_rad, img_rad + h_rad]).astype('int')
    sweep_bounds = np.array([img_rad - sweep_rad, img_rad + sweep_rad]).astype('int')

    # milliSecs per minute of movement
    movement_stop = 1 / speed  # mS / min

    n_movement_stops = sweep
    time_needed = movement_stop * n_movement_stops

    assert stim_space.shape[2] > time_needed, (
        f'Time required for sweep {sweep} is {time_needed},'
        f'which is more than total stim time {stim_space.shape[2]}'
    )

    # sweep + 1 , endpoint=True : purpose is to have, for every time_stop or frame, it's start and end
    # the +1 for sweep ensures that the first and second time_stops (idx 0,1) code the beg and end time for which
    # the bar is in the first position or minute.
    # second and third (idx 1,2) code beg and end time for time in second position / m inute
    # idx -2,-1 (second last and last) code time in last minute
    time_stops = np.linspace(0, time_needed, sweep + 1, endpoint=True).round().astype('int')

    # will have length of time_stops - 1 (due to time frame bracketing ... see above)
    sweep_stops = np.arange(sweep_bounds[0], sweep_bounds[1] + 1).astype('int')

    for t_idx in range(time_stops.size - 1):

        sweep_loc = sweep_stops[t_idx]
        time_beg, time_end = time_stops[[t_idx, t_idx + 1]]

        stim_space[
            h_bounds[0]: h_bounds[1] + 1,  # y
            sweep_loc - w_rad: sweep_loc + w_rad + 1,  # x
            time_beg: time_end
        ] = bar_lum

    return stim_space


def view_stim(stim, axes={'y': 0, 'x': 1, 't': 2}, xvals=None):
    #     if time_axis == 2:
    #         stim = stim.swapaxes(0,2).swapaxes(1,2)
    #     if time_axis == 1:
    #         stim = stim.swapaxes(0,1)

    if not xvals:
        xvals = np.arange(stim.shape[axes['t']])
    pg.image(stim, xvals=xvals, axes=axes)
    QtGui.QApplication.instance().exec_()


def mk_rf_spat_charact(rf, spat_freq, temp_freq, temp_res=10):
    '''Characterise response properties of RF by simple prod with sin grating

    Parameters
    ----
    rf : receptive field, output of mkRF
    spat_freq : spatial frequency of stimulus
        Passed to `spat_freq` arg of `mk_sinstim`
    temp_freq : temporal frequency of stimulus
        Passed to `temp_freq` arg of `mk_sinstim`
    temp_res : temporal resolution of stimulus
        Passed to `temp_res` of `mk_sinstim`


    Returns
    ----
    resp : array (1D)
        response of RF over temporal dimension of stimulusS
    resp_off : array (1D)
        Like resp, but inverted and translated by sum of rf
    stim : array (1D)
        Stimulus used ot generate response
        Output of mk_sinstim using spat_freq and temp_freq
    '''

    shape = rf.shape[0]
    # cent = shape//2

    stim = mk_sinstim(temp_freq=temp_freq, spat_freq=spat_freq, spat_ext=shape, temp_res=temp_res)

    rft = rf[..., np.newaxis] * np.ones(stim.shape[-1])

    resp = (rft * stim).sum(axis=(0, 1))
    resp_off = (-rft * stim).sum(axis=(0, 1)) + rf.sum()

    return resp, resp_off, stim


def plot_rf_spat_charact(rf, resp, resp_off, stim, figsize=(20, 7)):
    """Plot characterisation of rf resp

    Takes output/args of mk_rf_spat_charact

    Produces 3 subplots:

    * Stim as 1D slice
    * RF resp (putatively ON)
    * RF OFF resp (presuming RF is on)
    """

    shape = rf.shape[0]
    cent = shape // 2

    min_val, max_val = resp.min(), resp.max()
    min_val_off, max_val_off = resp_off.min(), resp_off.max()

    plt.figure(figsize=figsize)

    plt.subplot(1, 3, 1)
    plt.plot(stim[cent, cent, :])
    plt.title('stim')

    plt.subplot(1, 3, 2)
    plt.plot(resp)
    plt.axhline(min_val, ls=':', c='k')
    plt.axhline(max_val, ls=':', c='k')
    plt.axhline((max_val + min_val) / 2, ls=':', c='k')

    plt.title(f'sum: {rf.sum().round(4)}, offset: {np.mean([min_val, max_val]).round(4)}')

    plt.subplot(1, 3, 3)

    plt.plot(resp_off)
    plt.axhline(min_val_off, ls=':', c='k')
    plt.axhline(max_val_off, ls=':', c='k')
    plt.axhline((max_val_off + min_val_off) / 2, ls=':', c='k')

    plt.title(f'OFF sum: {rf.sum().round(4)}, offset: {np.mean([min_val_off, max_val_off]).round(4)}')

    plt.tight_layout()
