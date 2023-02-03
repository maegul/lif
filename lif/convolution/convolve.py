"""
Using receptive fields and stimuli to create firing rates through convolution
"""
from typing import Tuple, Union, Dict, Optional

from brian2.monitors.ratemonitor import PopulationRateMonitor
from brian2.monitors.spikemonitor import SpikeMonitor
import numpy as np
from scipy.signal import convolve

import brian2 as bn

from ..stimulus import stimulus as stim
from ..receptive_field.filters import filter_functions as ff

from ..utils import data_objects as do

from . import correction


def mk_single_sf_tf_response(
        sf: do.DOGSpatialFilter,
        tf: do.TQTempFilter,
        st_params: do.SpaceTimeParams,
        stim_params: do.GratingStimulusParams,
        stim_slice: np.ndarray,  # better object for this ... let's see what's needed?
        contrast_params: Optional[do.ContrastParams] = None,
        filter_actual_max_f1_amp: Optional[do.LGNF1AmpMaxValue] = None,
        target_max_f1_amp: Optional[do.LGNF1AmpMaxValue] = None,
        rectified: bool = True
        ) -> np.ndarray:
    """Produces rectified 1D response of "cell" defined by sf+tf to stim

    stim grating presumed to be 3D (temporal)

    Stimulus and filters are created to the same extent as defined in st_params

    Corrections for F1 and rectification are performed so that the rectified response
    will have an F1 and DC (from fourier analysis) as the filters dictate.
    """

    # Handle if max_f1 passed in or not
    if (
            (target_max_f1_amp or filter_actual_max_f1_amp) # at least one
            and not
            (target_max_f1_amp and filter_actual_max_f1_amp) # but not both
            ):
        raise ValueError('Need to pass BOTH target and actual max_f1_amp')

    xc, yc = ff.mk_spat_coords(st_params.spat_res, st_params.spat_ext)
    tc = ff.mk_temp_coords(st_params.temp_res, st_params.temp_ext)

    spat_filt = ff.mk_dog_sf(xc, yc, sf.parameters)
    temp_filt = ff.mk_tq_tf(tc, tf.parameters)

    # spatial convolution
    spatial_product = (spat_filt[..., np.newaxis] * stim_slice).sum(axis=(0, 1))

    # temporal convolution
    resp: np.ndarray = convolve(spatial_product, temp_filt)[:tc.value.size]

    # adjustment parameters for going from F1 SF and TF to convolution to accurate
    # sinusoidal response
    adj_params = correction.mk_conv_resp_adjustment_params(
        st_params, stim_params, sf, tf,
        contrast_params=contrast_params,
        filter_actual_max_f1_amp=filter_actual_max_f1_amp,
        target_max_f1_amp=target_max_f1_amp
        )

    true_resp = correction.adjust_conv_resp(resp, adj_params)

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


def aggregate_poisson_trials(
        spikes_record: Union[bn.SpikeMonitor, Dict, np.ndarray]
        ) -> np.ndarray:
    """Transform spike data into form appropriate for plotting

    2 Cols: [[number, spike_time], [number, spike_time], [...], ...]
    """

    cells, spike_times = [], []

    if isinstance(spikes_record, np.ndarray):
        spike_train_dict = {
            i: spikes_record[i,:]
                # presume each cell or trial is a row
                for i in range(spikes_record.shape[0])
            }
    elif isinstance(spikes_record, bn.SpikeMonitor):
            spike_train_dict = spikes_record.spike_trains()
    else:
        spike_train_dict = spikes_record


    for cell, spikes in spike_train_dict.items():
        # conversion to array necessary to take away brian cruft
        cells.append(np.ones_like(np.array(spikes)) * cell)
        spike_times.append(np.array(spikes))


    cells = np.concatenate(cells)
    spikes = np.concatenate(spike_times)

    all_spikes = np.stack([cells, spikes], axis=0)

    return all_spikes
