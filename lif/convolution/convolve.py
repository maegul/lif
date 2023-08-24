"""
Using receptive fields and stimuli to create firing rates through convolution
"""
from typing import Tuple, Union, Dict, Optional, Sequence, overload, cast
import itertools as it

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
    # faster way of doing manual convolution here than the older line below
    # `...` broadcasts additional dimensions
    spatial_product = np.einsum('ij,ij...', spat_filt, stim_slice)

    # older line
    # spatial_product = (spat_filt[..., np.newaxis] * stim_slice).sum(axis=(0, 1))

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
        log_print: bool = False, log_info: Optional[str] = None,
        ) -> bn.SpikeMonitor:
    """Generate Spike times for each array of continues rate times

    Each array in `response_arrays` is intended to represent a single cell

    `st_params` are necessary to know the temporal resolution and extent to simulate
    the poissonic generate of spikes
    """

    bn.start_scope()

    if log_print:
        print(f'{log_info} ... POISSON SPIKES')


    n_cells = len(response_arrays)
    rate_arrays = np.column_stack(response_arrays)

    if log_print:
        print(f'{log_info} ... POISSON SPIKES: shape of rate-arrays: {rate_arrays.shape}')

    timed_rate_arrays = bn.TimedArray(
        rate_arrays * bn.Hz,
        dt=st_params.temp_res.s * bn.second)

    if log_print:
        print(f'{log_info} ... POISSON SPIKES: shape timed_rate_arrays: {timed_rate_arrays.values.shape}')

    cells = bn.PoissonGroup(n_cells, rates='timed_rate_arrays(t, i)')
    spikes = bn.SpikeMonitor(cells)

    bn.run(st_params.temp_ext.s * bn.second)

    if log_print:
        print(f'{log_info} ... POISSON SPIKES: Run poisson spike generation')

    return spikes

    # number of brian clock steps per temporal res of sf_tf response
    # n_clock_steps = st_params.temp_res.s * bn.second / bn.defaultclock.dt

    # Turn response rate to brian array of rates, with appropriate time step

@overload
def mk_lgn_response_spikes(
        st_params: do.SpaceTimeParams,
        response_arrays: Tuple[np.ndarray, ...],
        n_trials: None = None,
        n_lgn_layers: None = None,
        n_inputs: None = None,
        log_print: bool = False, log_info: Optional[str] = None
        ) -> do.LGNLayerResponse: ...
@overload
def mk_lgn_response_spikes(
        st_params: do.SpaceTimeParams,
        response_arrays: Tuple[np.ndarray, ...],
        n_trials: int = 10,
        n_lgn_layers: None = None,
        n_inputs: None = None,
        log_print: bool = False, log_info: Optional[str] = None
        ) -> Tuple[do.LGNLayerResponse, ...]: ...
@overload
def mk_lgn_response_spikes(
        st_params: do.SpaceTimeParams,
        response_arrays: Tuple[np.ndarray, ...],
        n_trials: int,
        n_lgn_layers: int,
        n_inputs: Union[int, Sequence[int]],
        log_print: bool = False, log_info: Optional[str] = None
        ) -> Tuple[do.LGNLayerResponse, ...]: ...
def mk_lgn_response_spikes(
        st_params: do.SpaceTimeParams,
        response_arrays: Tuple[np.ndarray, ...],
        n_trials: Optional[int] = None,
        n_lgn_layers: Optional[int] = None,
        n_inputs: Optional[Union[int, Sequence[int]]] = None,
        log_print: bool = False, log_info: Optional[str] = None
        ) -> Union[do.LGNLayerResponse, Tuple[do.LGNLayerResponse, ...]]:
    """

    Must provide n_inputs if providing n_lgn_layers, as then the response arrays will be flattened
    and it cannot be determined form the number of resposne arrays how many input cells there are
    per lgn_layer without some calculation (IE, `len(response_arrays) / n_lgn_layers`) ...
    ... better to be explicit.
    """

    # Repeat responses for generating multiple trials of a single response

    # tile response arrays appropriately for repeated trials of the same responses
    if (n_trials is not None) and (n_lgn_layers is None):
        # repeated idxs (all cells x n_trials) ... eg (0, 1, 2, 0, 1, 2) (3 cells x 2 trials)
        repeated_cell_idxs = cells.mk_repeated_lgn_cell_idxs(n_trials=n_trials,
            n_cells=len(response_arrays)
            )
        response_arrays_for_spiking = tuple(
                response_arrays[i]
                for i in repeated_cell_idxs
            )
    # tile response arrays when multiple lgn layers and trials
    elif (
                (n_trials is not None) and (n_lgn_layers is not None)
                and (n_inputs is not None) and isinstance(n_inputs, int)
            ):
        # repeated idxs for trials but with multiple lgn layers, each with multiple cells
        # Thus ... cells x Trials x LGN Layers
        # EG (0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5) (3 cells x 2 trials x 2 lgn layers)
        # if
        repeated_cell_idxs = cells.mk_repeated_lgn_cell_idxs(
            n_trials=n_trials, n_cells=n_inputs,
            n_lgn_layers=n_lgn_layers)
        response_arrays_for_spiking = tuple(
                response_arrays[i]
                for i in repeated_cell_idxs
            )
    # tile response arrays when multiple layers and trials but each layer
    # has a variable amount of "overlapping regions" for synchrony
    elif (
                (n_trials is not None) and (n_lgn_layers is not None)
                # when layers have different "cells" because of synchrony and overlap maps
                and (n_inputs is not None) and isinstance(n_inputs, (tuple, list))
            ):

        if log_print:
            print(f'{log_info} ... LGN-SPIKES: Layers and Trials and overlapping regions')

        # n_cells = n_inputs
        # repeated idxs for trials but with multiple lgn layers, each with multiple cells
        # Thus ... cells x Trials x LGN Layers
        # EG (0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5) (3 cells x 2 trials x 2 lgn layers)
        # EXCEPT, when using synchrony and the number of "cells"/overlapping regions for each
        # ... layer is different.  Then, the same will be done but the sequence will be irregular
        # ... so as to reflect the different numbers of regions in each layer
        repeated_cell_idxs = cells.mk_repeated_lgn_cell_idxs(
            n_trials=n_trials, n_cells=n_inputs)
        response_arrays_for_spiking = tuple(
                response_arrays[i]
                for i in repeated_cell_idxs
            )

        if log_print:
            print(f'{log_info} ... LGN-SPIKES: Len response arrays: {len(response_arrays_for_spiking)}')

    else:
        response_arrays_for_spiking = response_arrays


    if log_print:
        print(f'{log_info} ... LGN-SPIKES: Actually creating spikes ...')


    spikes = mk_response_poisson_spikes(st_params, response_arrays_for_spiking)

    if log_print:
        print(f'{log_info} ... LGN-SPIKES: CREATED SPIKES!...')

    spike_times = spikes.spike_trains()

    # convert to native object type (native to this code base)
    all_cell_spike_times: Sequence[Time[np.ndarray]] = []
    cell_indices = list(spike_times.keys())

    # convert to aboslute time values (as will be a Time quantity)
    # Use seconds to convert and then to store
    all_cell_spike_times = tuple(
            Time(spike_times[ci] / bn.second, 's')
            for ci in cell_indices
        )

    if log_print:
        print(f'{log_info} ... LGN-SPIKES: Number spike time arrays: {len(all_cell_spike_times)}')


    # older way ... normal for loop
    # for ci in cell_indices:
         # convert to aboslute time values (as will be a Time quantity)
         # Use seconds to convert and then to store
    #     cell_spike_times: Time[np.ndarray] = Time(spike_times[ci] / bn.second, 's')
    #     all_cell_spike_times.append(cell_spike_times)
    # all_cell_spike_times = tuple(all_cell_spike_times)

    # Create response objects
    # Trials but not multiple sims/lgn layers
    if (n_trials is not None) and (n_lgn_layers is None):
        n_cells = len(response_arrays)
        # make a separate lgn response object for each trial
        # IE, each ojbect has responses of all cells' of the lgn layer, but for only trial
        lgn_response = tuple(
                do.LGNLayerResponse(
                    cell_spike_times = all_cell_spike_times[
                        # take all cells of a single trial, where `trial` increments
                        # which set of cells are taken by multiplying `trial` by the number of cells
                        0 + (n_cells * trial) : n_cells + (n_cells * trial)
                        ],
                    cell_rates = response_arrays
                )
                for trial in range(n_trials)
            )

    # trials and multiple lgn layers / sims
    elif (not (n_trials is None)) and (not (n_lgn_layers is None)) and isinstance(n_inputs, int):
        n_cells = n_inputs
        # create a flattened tuple in same structure as input response arrays
        # each response object is for a single trial for a single lgn layer
        # eg (layer1-trial1, layer1-trial2, layer2-trial1, layer2-trial2) (2 trials x 2 layers)
        lgn_response = tuple(
                do.LGNLayerResponse(
                    cell_spike_times = all_cell_spike_times[
                        # take all cells of a trial, incrementing through each set of cells
                        # by multiplying `trial` by `n_cells` as above.
                        # In this case, though, each trial is for a particular lgn layer
                        0 + (n_cells * trial) : n_cells + (n_cells * trial)
                    ],
                    cell_rates = response_arrays[
                        # Only need response arrays for the layer these responses are coming from
                        # find lgn layer by integer dividing `trial` by `n_trials`
                        # then use n lgn layer to increment over the set of cells using `n_cells`
                        # as the response arrays are (n_cells x n_layers)
                        # eg (cell1-layer1, cell2-layer1, cell1-layer2, cell2-layer2, ...)
                        # Basically, wait for `trial` to get through all the trials of a single
                        # lgn layer then increment up to the next lgn layer:
                        # `trial//n_trials = n_lgn_layer`
                        0 + (n_cells * (trial//n_trials)) : n_cells + (n_cells * (trial//n_trials))
                    ]
                )
                # generically, there is an lgn response for each layer *and* trial, so each
                #   trial for each layer is a separate response here ... go through each one
                for trial in range(n_lgn_layers * n_trials)
            )

    # variable number of cells per layer (ie for synchrony)
    elif (
                (not (n_trials is None)) and (not (n_lgn_layers is None))
                and isinstance(n_inputs, (tuple, list))
            ):

        if log_print:
            print(f'{log_info} ... LGN-SPIKES: Preparing response objects (for synchrony)')


        # Collect spike times into separate trial-layers

        # each pair of values will be the start and end idxs of slices that will provide each
        # trial-layer subset of arrays
        # 1. take the number of cells in each layer from n_inputs and repeat this number n_trials
        # 2. Flatten each of these repeats of the number into a single iterable
        # 3. Perform a cumulative sum over all these numbers,
        # 4. starting the cumulative sum from 0 so that the first slice can start at the beginning
        # Result is iterable of numbers, each number representing the number of cells
        # that make up a single trial-layer.
        # Eg, for n_inputs (5, 7, 3) and n_trials 2: (0, 5, 10, 17, 24, 27, 30)
        # for (0,5), (5, 10), (10, 17), ...

        trial_idxs = tuple(
            it.accumulate(                             # 3
                it.chain.from_iterable(                # 2
                        it.repeat(n_cells, n_trials)   # 1
                        for n_cells in n_inputs
                    ),                                 # 4
                initial=0
                )
            )

        if log_print:
            print(f'{log_info} ... LGN-SPIKES: number of trial_idxs: {len(trial_idxs)}')


        # Make iterable (generator) of each trial-layer by slicing all spike times using
        # values calculated above
        cell_spike_times = (
                all_cell_spike_times[a : b]
                for a,b in zip(
                        trial_idxs[:-1],
                        trial_idxs[1:]
                    )
            )

        # Collect response arrays into separate trial-layers to pair up with spike times above

        # Slicing indices for each layer without trials
        resp_rate_idxs_base = tuple(it.accumulate(n_inputs, initial=0) )

        # take all start indices (of the slice for each layer) and repeat for all trials
        starts = tuple(it.chain.from_iterable(
                it.repeat(rr_idx, n_trials)
                for rr_idx in resp_rate_idxs_base[:-1]
            ))

        # take all end indices (of the slice for each layer) and repeat for all trials
        ends = tuple(it.chain.from_iterable(
                it.repeat(rr_idx, n_trials)
                for rr_idx in resp_rate_idxs_base[1:]
            ))

        # Generator of each trial-layer response array (should pair up with spike time arrays above)
        cell_response_arrays = (
                response_arrays[a : b]
                for a,b in zip(starts, ends)
            )

        # All lgn responses
        lgn_response = tuple(
                do.LGNLayerResponse(cell_spike_times = spike_times, cell_rates = responses)
                    for spike_times, responses
                    in zip(cell_spike_times, cell_response_arrays)
            )


        if log_print:
            print(f'{log_info} ... LGN-SPIKES: Number of resposne objects: {len(lgn_response)}')

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
