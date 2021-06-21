"""Module for classes and functions for creating a Leaky Integrate and Fire model


"""

import numpy as np
from scipy.signal import fftconvolve as conv
import matplotlib.pyplot as plt

import brian2 as bn
# import brian2tools as bnt

# import receptive_field.rf_stim


def single_rf_resp(
        stim,
        cent, surr,
        tf=None, cent_tf=None, surr_tf=None,
        off_cell=False,
        return_cent_surr=False):
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

    rf_sz = cent.shape[0]
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
    cent_stim_prod = (cent[..., np.newaxis] * stim).sum(axis=(0, 1))  # create multiplicative response
    surr_stim_prod = (surr[..., np.newaxis] * stim).sum(axis=(0, 1))  # create multiplicative response

    # Temporal RF Response

    # Sorting out temp filt args and types
    # Must have at least a single tf or two in cent_tf and surr_tf

    if tf is None:  # not single tf provided
        # assert cent_tf and surr_tf are arrays
        assert isinstance(cent_tf, np.ndarray) and isinstance(surr_tf, np.ndarray), (
            'If tf is None, cent_tf and surr_tf must be provided',
            f'Here, cent_tf : {type(cent_tf)}, surr_tf : {type(surr_tf)}')

        cent_resp = conv(cent_stim_prod, cent_tf, mode='full')
        surr_resp = conv(surr_stim_prod, surr_tf, mode='full')

        full_resp = cent_resp - surr_resp

    else:  # single tf is provided
        assert isinstance(tf, np.ndarray), f'tf must be ndarray, instead is {type(tf)}'

        full_resp = conv(cent_stim_prod - surr_stim_prod, tf, mode='full')

    # sorting out ouputs and making OFF cell
    if off_cell:
        resp = -full_resp + (cent - surr).sum()  # negative of on_rf resp plus sum of on_rf
    else:
        resp = full_resp

    resp = resp[:stim.shape[-1]]  # take only length of stim to cut off tail of convolution

    if return_cent_surr:
        return resp, cent_resp, surr_resp
    else:
        return resp


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


def plot_spikes(resp, spikemon, popspikemon, figsize=(12, 6)):

    sim_size = resp.size

    plt.figure(figsize=figsize)

    plt.subplot(211)
    bnt.brian_plot(spikemon, markersize=1)
    plt.xlim(right=sim_size)

    plt.subplot(212)
    bnt.plot_rate(popspikemon.t, popspikemon.smooth_rate('gaussian', width=0.01 * bn.second), lw=5)
    plt.xlim(right=sim_size)
