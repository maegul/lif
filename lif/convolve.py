"""
Using receptive fields and stimuli to create firing rates through convolution
"""
from typing import Tuple

from brian2.monitors.ratemonitor import PopulationRateMonitor
from brian2.monitors.spikemonitor import SpikeMonitor
import numpy as np
from scipy.signal import convolve

import brian2 as bn

from .stimulus import stimulus as stim
from .receptive_field.filters import filter_functions as ff

from .utils import data_objects as do


def mk_single_sf_tf_response(
        sf: do.DOGSpatialFilter,
        tf: do.TQTempFilter,
        st_params: do.SpaceTimeParams,
        stim_params: do.GratingStimulusParams,
        rectified: bool = True
        ) -> np.ndarray:
    """Produces rectified 1D response of "cell" defined by sf+tf to stim

    stim grating presumed to be 3D (temporal)

    Stimulus and filters are created to the same extent as defined in st_params
    """

    xc, yc = ff.mk_spat_coords(st_params.spat_res, st_params.spat_ext)
    tc = ff.mk_temp_coords(st_params.temp_res, st_params.temp_ext)

    spat_filt = ff.mk_dog_sf(xc, yc, sf.parameters)
    temp_filt = ff.mk_tq_tf(tc, tf.parameters)

    grating = stim.mk_sinstim(st_params, stim_params)

    spatial_product = (spat_filt[..., np.newaxis] * grating).sum(axis=(0, 1))

    resp: np.ndarray = convolve(spatial_product, temp_filt)[:tc.value.size]

    adj_params = ff.mk_conv_resp_adjustment_params(
        st_params, stim_params, sf, tf)

    true_resp = ff.adjust_conv_resp(resp, adj_params)

    if rectified:
        true_resp[true_resp < 0] = 0

    return true_resp


def mk_sf_tf_poisson(
        st_params: do.SpaceTimeParams,
        resp: np.ndarray, n_trials: int = 20
        ) -> Tuple[bn.SpikeMonitor, bn.PopulationRateMonitor]:

    bn.start_scope()

    # number of brian clock steps per temporal res of sf_tf response
    # n_clock_steps = st_params.temp_res.s * bn.second / bn.defaultclock.dt

    # Turn response rate to brian array of rates, with appropriate time step
    I_recorded = bn.TimedArray(  # noqa: F841
        resp * bn.Hz, dt=st_params.temp_res.s * bn.second)  # type: ignore

    cell = bn.PoissonGroup(n_trials, rates='I_recorded(t)')
    spikes = bn.SpikeMonitor(cell)
    pop_spikes = bn.PopulationRateMonitor(cell)

    bn.run(st_params.temp_ext.s * bn.second)  # type: ignore

    return spikes, pop_spikes


def aggregate_poisson_trials(spikes_record: bn.SpikeMonitor) -> np.ndarray:

    cells, spike_times = [], []

    for cell, spikes in spikes_record.spike_trains().items():
        # conversion to array necessary to take away brian cruft
        cells.append(np.ones_like(np.array(spikes)) * cell)
        spike_times.append(np.array(spikes))

    cells = np.concatenate(cells)
    spikes = np.concatenate(spike_times)

    all_spikes = np.stack([cells, spikes], axis=0)

    return all_spikes
