"""
Using receptive fields and stimuli to create firing rates through convolution
"""
from typing import Tuple, Union, Dict, Optional, Sequence, overload, cast

from brian2.monitors.ratemonitor import PopulationRateMonitor
from brian2.monitors.spikemonitor import SpikeMonitor
import numpy as np
from scipy.signal import convolve

import brian2 as bn

from ..stimulus import stimulus as stim
from ..receptive_field.filters import filter_functions as ff
from ..lgn import cells

from ..utils import data_objects as do, exceptions as exc
from ..utils.units import Time

from . import correction


def mk_single_sf_tf_response(
        st_params: do.SpaceTimeParams,
        sf: do.DOGSpatialFilter,
        tf: do.TQTempFilter,
        spat_filt: np.ndarray, temp_filt: np.ndarray,
        stim_params: do.GratingStimulusParams,
        stim_slice: np.ndarray,  # better object for this ... let's see what's needed?
        contrast_params: Optional[do.ContrastParams] = None,
        filter_actual_max_f1_amp: Optional[do.LGNF1AmpMaxValue] = None,
        target_max_f1_amp: Optional[do.LGNF1AmpMaxValue] = None,
        rectified: bool = True
        ) -> do.ConvolutionResponse:
    """Produces rectified 1D response of "cell" defined by sf+tf to stim

    stim grating presumed to be 3D (temporal)

    Stimulus and filters are created to the same extent as defined in st_params

    Corrections for F1 and rectification are performed so that the rectified response
    will have an F1 and DC (from fourier analysis) as the filters dictate.
    """

    # # Handle if max_f1 passed in or not
    if (
            (target_max_f1_amp or filter_actual_max_f1_amp) # at least one
            and not
            (target_max_f1_amp and filter_actual_max_f1_amp) # but not both
            ):
        raise ValueError('Need to pass BOTH target and actual max_f1_amp')

    # # requrie that the spatial filter and the stimulus slice are the same size
    # stim_slice also has temporal dimension (3rd), so take only first two
    if not (stim_slice.shape[:2] == spat_filt.shape):
        raise exc.LGNError('Stimulus slice and spatial filter array are not the same shape')

    # # spatial convolution
    spatial_product = (spat_filt[..., np.newaxis] * stim_slice).sum(axis=(0, 1))

    # # temporal convolution

    # prepare temp buffer
    # Doesn't actually help get a stable sinusoidal response
    # ... leaving here just in case it's useful later
    # sf_conv_amp = correction.mk_dog_sf_conv_amp(
    #     freqs_x=stim_params.spat_freq_x,
    #     freqs_y=stim_params.spat_freq_y,
    #     dog_args=sf.parameters, spat_res=st_params.spat_res
    #     )
    # buffer_val = sf_conv_amp * stim_params.DC
    # print(sf_conv_amp, buffer_val)
    # temp_res_unit = st_params.temp_res.unit
    # buffer_size = int(Time(200, 'ms')[temp_res_unit] / st_params.temp_res.value ) + 1
    # buffer = np.ones(buffer_size) * buffer_val

    # spatial_product_w_buffer = np.r_[buffer, spatial_product]
    # take temporal extent of stimulus, as convolve will go to extent of stim+temp_filt
    # resp: np.ndarray = convolve(
    #     spatial_product_w_buffer, temp_filt
    #     )[buffer_size : (stim_slice.shape[2]+buffer_size)]

    resp: np.ndarray = convolve(spatial_product, temp_filt)[:stim_slice.shape[2]]

    # # adjustment parameters
    # for going from F1 SF and TF to convolution to accurate sinusoidal response
    adj_params = correction.mk_conv_resp_adjustment_params(
        st_params, stim_params, sf, tf,
        contrast_params=contrast_params,
        filter_actual_max_f1_amp=filter_actual_max_f1_amp,
        target_max_f1_amp=target_max_f1_amp
        )

    # # apply adjustment
    true_resp = correction.adjust_conv_resp(resp, adj_params)

    # # recification
    if rectified:
        true_resp[true_resp < 0] = 0

    return do.ConvolutionResponse(response=true_resp, adjustment_params=adj_params)


def mk_response_poisson_spikes(
        st_params: do.SpaceTimeParams,
        response_arrays: Tuple[np.ndarray, ...],
        ) -> bn.SpikeMonitor:
    """Generate Spike times for each array of continues rate times

    Each array in `response_arrays` is intended to represent a single cell

    `st_params` are necessary to know the temporal resolution and extent to simulate
    the poissonic generate of spikes
    """

    bn.start_scope()

    n_cells = len(response_arrays)
    rate_arrays = np.column_stack(response_arrays)
    timed_rate_arrays = bn.TimedArray(
        rate_arrays * bn.Hz,
        dt=st_params.temp_res.s * bn.second)

    cells = bn.PoissonGroup(n_cells, rates='timed_rate_arrays(t, i)')
    spikes = bn.SpikeMonitor(cells)

    bn.run(st_params.temp_ext.s * bn.second)

    return spikes

    # number of brian clock steps per temporal res of sf_tf response
    # n_clock_steps = st_params.temp_res.s * bn.second / bn.defaultclock.dt

    # Turn response rate to brian array of rates, with appropriate time step

@overload
def mk_lgn_response_spikes(
        st_params: do.SpaceTimeParams,
        response_arrays: Tuple[np.ndarray, ...],
        n_trials: None = None
        ) -> do.LGNLayerResponse: ...
@overload
def mk_lgn_response_spikes(
        st_params: do.SpaceTimeParams,
        response_arrays: Tuple[np.ndarray, ...],
        n_trials: int = 10
        ) -> Tuple[do.LGNLayerResponse, ...]: ...
def mk_lgn_response_spikes(
        st_params: do.SpaceTimeParams,
        response_arrays: Tuple[np.ndarray, ...],
        n_trials: Optional[int] = None
        ) -> Union[do.LGNLayerResponse, Tuple[do.LGNLayerResponse, ...]]:


    n_cells = len(response_arrays)
    if n_trials is not None:
        # repeated idxs (all cells x n_trials) ... eg (0, 1, 2, 0, 1, 2) (3 cells x 2 trials)
        repeated_cell_idxs = cells.mk_repeated_lgn_cell_idxs(n_trials=n_trials, n_cells=n_cells)
        response_arrays_for_spiking = tuple(
                response_arrays[i]
                for i in repeated_cell_idxs
            )
    else:
        response_arrays_for_spiking = response_arrays

    spikes = mk_response_poisson_spikes(st_params, response_arrays_for_spiking)
    spike_times = spikes.spike_trains()

    # convert to native object type (native to this code base)
    all_cell_spike_times: Sequence[Time[np.ndarray]] = []
    cell_indices = list(spike_times.keys())

    for ci in cell_indices:
        # convert to aboslute time values (as will be a Time quantity)
        # Use seconds to convert and then to store
        cell_spike_times: Time[np.ndarray] = Time(spike_times[ci] / bn.second, 's')
        all_cell_spike_times.append(cell_spike_times)

    all_cell_spike_times = tuple(all_cell_spike_times)

    if n_trials is not None:
        # make a separate lgn response object for each trial
        # separate response object for each trial
        lgn_response = tuple(
                do.LGNLayerResponse(
                    cell_spike_times = all_cell_spike_times[
                        0 + (n_cells * trial) : n_cells + (n_cells * trial)
                        ],
                    cell_rates = response_arrays
                )
                for trial in range(n_trials)
            )
    else:
        lgn_response = do.LGNLayerResponse(
                cell_rates = response_arrays,
                cell_spike_times = all_cell_spike_times
            )

    return lgn_response




def mk_sf_tf_poisson(
        st_params: do.SpaceTimeParams,
        resp: np.ndarray,
        n_trials: int = 20
        ) -> Tuple[bn.SpikeMonitor, bn.PopulationRateMonitor]:

    bn.start_scope()

    # number of brian clock steps per temporal res of sf_tf response
    # n_clock_steps = st_params.temp_res.s * bn.second / bn.defaultclock.dt

    # Turn response rate to brian array of rates, with appropriate time step
    I_recorded = bn.TimedArray(  # noqa: F841
        resp * bn.Hz,
        dt=st_params.temp_res.s * bn.second  # type: ignore
        )

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
