# ===========
import numpy as np
import numpy as np
from scipy import sparse
from scipy.signal import fftconvolve as conv
# import matplotlib.pyplot as plt

import brian2 as bn

from lif import *
# -----------
# ===========
import plotly.express as px
# -----------
# ===========
def mk_sinstim(
        ori=90, spat_freq=1, temp_freq: TempFrequency = TempFrequency(1),
        spat_ext=ArcLength(120, 'mnt'), temp_ext=Time(1000, 'ms'),
        spat_res=ArcLength(1, 'mnt'), temp_res=Time(1, 'ms')):
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
    ori = np.radians(ori + 90)  # type: ignore

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
    # omega = temp_freq * 2 * np.pi

    # creating coords
    xc, yc, tc = mk_spat_temp_coords(
        spat_ext=spat_ext, temp_ext=temp_ext,
        spat_res=spat_res, temp_res=temp_res
        )

    # Convert spatial coords to degs (from minutes)
    # Convert temporal coords to secs (from milliSecs)
    # scale coords so that unit -> spat_freq, temp_freq cycles
    # xc = kx * (xc / 60)
    # yc = ky * (yc / 60)
    # tc = omega * tc / 1000

    img = 0.5 + 0.5 * np.cos(kx*xc.deg + ky*yc.deg + (temp_freq.w * tc.s))

    return img

# -----------
# ===========
# img = mk_sinstim()
# -----------
# ===========
# img.shape
# -----------
# ===========
# px.imshow(img[...,500]).show()
# -----------
# ===========
def single_rf_resp(
        stim,
        rf,
        # cent, surr,
        tf,
        # cent_tf=None, surr_tf=None,
        # off_cell=False,
        # return_cent_surr=False
        ):
    """Generate the response of a single RF to a stimulus

    RF is centered in the stimulus

    Parameters
    ----
    stim : array (2d)
        Stimulus

    cent, surr
        Center and surround of RF (spatial)

    tf : None, array (1d)
        temporal filter for when single temp filter being used
        If None (default), must provide `cent_tf` and `surr_tf`

    cent_tf, surr_tf : None, array (1D)
        temporal filters for cent and surr components separately
        If None, must provide `tf`.

    """

    rf_sz = rf.shape[0]
    stim_sz = stim.shape[0]

    # stim must be equal to or bigger than rf
    assert stim_sz >= rf_sz, f'Stim must not be smaller than rf in size'
    # If stim bigger, must be odd and with a center

    if stim_sz > rf_sz:
        assert (stim_sz % 2 == 1), f'stim size is not odd and has no center'
        assert (rf_sz % 2 == 1), f'rf size is not odd and has no center'

        stim_cent = stim_sz // 2
        rf_breadth = rf_sz // 2

        # Take slice out of stim, at center, same size as rf
        # Reassign slice to stim

        # add 1 to get full rf_breadth for end of slice (because of python)
        stim_rf_coords = stim_cent - rf_breadth, stim_cent + rf_breadth + 1
        stim = stim[stim_rf_coords[0]:stim_rf_coords[1], stim_rf_coords[0]:stim_rf_coords[1]]

    # Spatial RF response
    rf_stim_prod = (rf[..., np.newaxis] * stim).sum(axis=(0, 1))  # create multiplicative response
    # cent_stim_prod = (cent[..., np.newaxis] * stim).sum(axis=(0, 1))  # create multiplicative response
    # surr_stim_prod = (surr[..., np.newaxis] * stim).sum(axis=(0, 1))  # create multiplicative response


    assert isinstance(tf, np.ndarray), f'tf must be ndarray, instead is {type(tf)}'

    full_resp = conv(rf_stim_prod, tf, mode='full')

    # sorting out ouputs and making OFF cell
    # if off_cell:
    #     resp = -full_resp + (cent - surr).sum()  # negative of on_rf resp plus sum of on_rf
    # else:
    #     resp = full_resp

    resp = full_resp
    resp = resp[:stim.shape[-1]]  # take only length of stim to cut off tail of convolution

    # if return_cent_surr:
    #     return resp, cent_resp, surr_resp
    # else:
    #     return resp
    return resp

# -----------
# ===========
tf = TQTempFilter.load(TQTempFilter.get_saved_filters()[0])
sf = DOGSpatialFilter.load(DOGSpatialFilter.get_saved_filters()[0])
# -----------
# ===========
stim_amp=0.5
spat_res=ArcLength(1, 'mnt')
spat_ext=ArcLength(120, 'mnt')
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')
temp_freq = TempFrequency(4)
spat_freq_x = SpatFrequency(2)
spat_freq_y = SpatFrequency(0)
# -----------
# ===========
conv_fact = conv_amp_adjustment_factor(
    stim_amp=stim_amp, spat_res=spat_res, temp_res=temp_res,
    tf=tf, sf=sf, temp_freqs=temp_freq,
    spat_freqs_x=spat_freq_x,
    spat_freqs_y=spat_freq_y
    )
# -----------
# ===========
conv_fact
# -----------
# ===========
xc_spat, yc_spat = mk_spat_coords(
    spat_ext=spat_ext, spat_res=spat_res)
# -----------
# ===========
xc_spat.mnt.shape
# -----------
# ===========
tc_temp = mk_temp_coords(temp_res=temp_res, temp_ext=temp_ext)
# -----------
# ===========
tc_temp.s.shape
# -----------
# ===========
# xc, yc, tc = mk_spat_temp_coords(
#         spat_ext=ArcLength(120, 'mnt'), temp_ext=Time(1000, 'ms'),
#         spat_res=ArcLength(1, 'mnt'), temp_res=Time(1, 'ms')
#         )
# -----------
# ===========
# xc.mnt.shape
# -----------
# ===========
rf = mk_dog_sf(x_coords=xc_spat, y_coords=yc_spat, dog_args=sf.parameters)
tfilt = mk_tq_tf(t=tc_temp, tf_params=tf.parameters)
# -----------
# ===========
rf.shape, tfilt.shape
# -----------
# ===========
px.imshow(rf).show()
# -----------
# ===========
img = mk_sinstim(spat_freq=2, temp_freq=TempFrequency(4))
# -----------
# ===========
img.shape
# -----------
# ===========
img.max(), img.min()
# -----------
# ===========
px.imshow(img[...,500]).show()
# -----------

# Custom Convolution
# ===========

stim_sz = img.shape[0]
rf_sz = rf.shape[0]

# img must be equal to or bigger than rf
assert stim_sz >= rf_sz, f'Stim must not be smaller than rf in size'
# If img bigger, must be odd and with a center

# -----------
# ===========
if stim_sz > rf_sz:
    assert (stim_sz % 2 == 1), f'stim size is not odd and has no center'
    assert (rf_sz % 2 == 1), f'rf size is not odd and has no center'

    stim_cent = stim_sz // 2
    rf_breadth = rf_sz // 2

    # Take slice out of stim, at center, same size as rf
    # Reassign slice to stim

    # add 1 to get full rf_breadth for end of slice (because of python)
    stim_rf_coords = stim_cent - rf_breadth, stim_cent + rf_breadth + 1
    stim = img[stim_rf_coords[0]:stim_rf_coords[1], stim_rf_coords[0]:stim_rf_coords[1]]
else:
    stim = img
# -----------
# ===========
temp_prod = rf[..., np.newaxis] * stim
# -----------
# ===========
temp_prod.shape
# -----------
# ===========
px.imshow(temp_prod[..., 100]).show()
# -----------
# ===========
rf_stim_prod = (rf[..., np.newaxis] * stim).sum(axis=(0, 1))  # create multiplicative response
# -----------
# ===========
mk_dog_sf_conv_amp(SpatFrequency(2), SpatFrequency(0), dog_args=sf.parameters, spat_res=spat_res)
# -----------
# ===========
mk_dog_sf_conv_amp(
    SpatFrequency(2), SpatFrequency(0),
    dog_args=sf.parameters, spat_res=spat_res
    ) / spat_res.deg
# -----------
# ===========
mk_dog_sf_ft(SpatFrequency(2), SpatFrequency(0), dog_args=sf.parameters)
# -----------
# ===========
rf_stim_prod.max() - rf_stim_prod.min()
# -----------
# ===========
conv()
# -----------
# ===========
px.line(rf_stim_prod).show()
# -----------
# ===========
resp = single_rf_resp(stim=img, rf=rf, tf=tfilt)
# -----------
# ===========
resp.shape
# -----------
# ===========
resp.max() * conv_fact
# -----------
# ===========
px.line(y=resp).show()
# -----------
# ===========
px.line(x=tc_temp.s, y=resp*conv_fact).show()
# -----------
# ===========
1.8*1000
# -----------
# ===========
def gen_poisson_spikes(resp, rate_scale=1, n_cells=1):

    sim_time = resp.size / 1000 * bn.second  # all time is in milliseconds
    # number of time steps in simulation given the length of the response

    # all temporal params are in milliseconds
    dt = (1 / 1000) * bn.second

    spike_rate = bn.TimedArray(rate_scale * resp * bn.Hz, dt=dt)

    P = bn.PoissonGroup(N=n_cells, rates='spike_rate(t)')
    S = bn.SpikeMonitor(P)
    M = bn.PopulationRateMonitor(P)

    print('running sim')
    bn.run(sim_time)

    return S, M
# -----------
# ===========
s,m = gen_poisson_spikes(resp*conv_fact)
# -----------
# ===========

# -----------
# ===========
mk_tq_tf_conv_amp(TempFrequency(4), tf.parameters,Time(1, 'ms'))
# -----------
# ===========
mk_tq_tf_ft(TempFrequency(4), tf.parameters)
# -----------

