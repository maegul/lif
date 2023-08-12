"""Create responses of a LGN layer to a stimulus

"""

import math
import datetime as dt
import re
from textwrap import dedent
from typing import Any, List, Sequence, Dict, Tuple, Optional, Union, cast, overload
from pathlib import Path
import shutil
import datetime as dt
import pickle
import itertools

import numpy as np

from ..utils import data_objects as do, exceptions as exc
from lif.utils.units.units import (
        ArcLength, Time, SpatFrequency, TempFrequency
    )

from ..receptive_field.filters import filter_functions as ff

from ..convolution import (
    convolve,
    soodak_rf as srf,
    correction
    )
from ..lgn import (
    cells,
    spat_filt_overlap as sfo
    )
from ..stimulus import stimulus
from . import (
    all_filter_actual_max_f1_amp as all_max_f1,
    leaky_int_fire as lif_model
    )
from lif import lgn


# def mk_lgn_layer_responses(
#         lgn_params: do.LGNParams,
#         stim_params: do.GratingStimulusParams,
#         space_time_params: do.SpaceTimeParams
#         ):
#     """For LGN layer and stimulus parameters, return firing rate responses
#     """

#     lgn_layer = cells.mk_lgn_layer(lgn_params, space_time_params.spat_res)


# # Utility Functions for running simulations

def create_stimulus(
        params: do.SimulationParams,
        force_central_rf_locations: bool,
        ) -> Tuple[do.GratingStimulusParams]:

    # check that st_params spat_ext is sufficient (based on worst case extent)
    # ... only one space_time params for all the stimuli ... so only check once
    if not force_central_rf_locations:
        estimated_max_spat_ext = stimulus.estimate_max_stimulus_spatial_ext_for_lgn(
            params.space_time_params.spat_res, params.lgn_params,
            n_cells=2000, safety_margin_increment=0.1
            )
    else:
        # just use largest spatial filter as max required extent
        estimated_max_spat_ext = cells.calculate_max_spatial_ext_of_all_spatial_filters(
                spat_res=params.space_time_params.spat_res,
                lgn_params=params.lgn_params,
                safety_margin_increment=0  # shouldn't be necessary if they're centered
            )
    print(dedent(
        f'''\
        --------
        Stimulus
        --------

        Estimated max spatial extent: {estimated_max_spat_ext.mnt:.3f} mnts
        Current spatial extent:       {params.space_time_params.spat_ext.mnt:.3f} mnts
        ''')
    )

    if (params.space_time_params.spat_ext.base < estimated_max_spat_ext.base):
        raise exc.LGNError(dedent(
            f'''\
            Spatial extent of space time params ({params.space_time_params.spat_ext.mnt: .3f} mnts)
            is less than estimated required maximum ({estimated_max_spat_ext.mnt:.3f} mnts)
            '''))

    # create all Stimuli necessary

    # Generate all parameters (ie, all combinations of desired parameters)
    all_stim_params = stimulus.mk_multi_stimulus_params(params.multi_stim_params)

    # Create and save all stimuli arrays if not already generated and saved to file
    stimulus.mk_stimulus_cache(params.space_time_params, all_stim_params)

    return all_stim_params



def create_all_lgn_layers(
        params: do.SimulationParams,
        force_central_rf_locations: bool,
        ) -> do.ContrastLgnLayerCollection:

    # As lgn layer depends on contrast, handle whether multiple contrasts set or just default
    #                        V--> Can use ContrastValue as key as it's frozen
    all_lgn_layers: Dict[do.ContrastValue, Tuple[do.LGNLayer]]  # key is contrast

    # need stim contrasts for creating max amplitude corrections in the lgn layer
    # So manage if there are multiple contrasts in the multi stim params
    multi_stim_contrasts = params.multi_stim_params.contrasts

    print(dedent(
        f'''\
        --------
        LGN Layers
        --------

        Making all LGN layers: {params.n_simulations} layers
        ''') )

    # none defined in multi stim, so use default from grating stim class
    # ... should really have used stim settings for a default contrast ... oh well!!
    if multi_stim_contrasts is None:
        stim_contrast: do.ContrastValue = (
            do.GratingStimulusParams.__dataclass_fields__['contrast'].default )
        all_lgn_layers = {
            stim_contrast:
            tuple(
                cells.mk_lgn_layer(
                    params.lgn_params,
                    spat_res=params.space_time_params.spat_res,
                    contrast=stim_contrast,
                    force_central=force_central_rf_locations)
                for _ in range(params.n_simulations)
            )
        }
    # If contrasts are defined, create an LGN layer for each
    # (shouldn't be too much memory as each LGN layer isn't too big)
    # elif multi_stim_contrasts is not None:
    else:
        all_stim_contrasts = tuple(do.ContrastValue(c) for c in multi_stim_contrasts)

        all_lgn_layers = {
            stim_contrast:
            tuple(
                cells.mk_lgn_layer(
                    params.lgn_params,
                    spat_res=params.space_time_params.spat_res,
                    contrast=stim_contrast,
                    force_central=force_central_rf_locations)
                for _ in range(params.n_simulations)
            )
            for stim_contrast in all_stim_contrasts
        }

    # maybe save first?

    return all_lgn_layers


def create_all_lgn_layer_overlapping_regions(
        all_lgn_layers: do.ContrastLgnLayerCollection,
        params: do.SimulationParams
        ) -> Dict[do.ContrastValue, Tuple[sfo.LGNOverlapMap, ...]]:

    # key is contrast
    all_layers_overlaps = {
        contrast: tuple(
                    sfo.mk_lgn_overlapping_weights_vect(lgn_layer, params.space_time_params)
                        for lgn_layer in lgn_layers
                )
        for contrast, lgn_layers in all_lgn_layers.items()
    }

    return all_layers_overlaps


def mk_adjusted_overlapping_regions_wts(
        all_lgn_overlapping_maps: Dict[do.ContrastValue, Tuple[sfo.LGNOverlapMap, ...]]
        ) -> Dict[do.ContrastValue, Tuple[sfo.LGNOverlapMap, ...]]:

    all_adjusted_overlap_maps = dict()

    for contrast_value, overlap_maps in all_lgn_overlapping_maps.items():
        adjusted_overlap_maps = tuple(
                    {
                    cell_idxs: {
                        cell_idx: cell_wt / len(cell_idxs)
                        for cell_idx, cell_wt in cell_wts.items()
                    }
                    for cell_idxs, cell_wts in overlap_map.items()
                }
                for overlap_map in overlap_maps
            )

        all_adjusted_overlap_maps[contrast_value] = adjusted_overlap_maps

    return all_adjusted_overlap_maps


def create_all_lgn_layer_adjusted_overlapping_regions(
        all_lgn_layers: do.ContrastLgnLayerCollection,
        params: do.SimulationParams
        ) -> Dict[do.ContrastValue, Tuple[sfo.LGNOverlapMap, ...]]:

    all_overlap_maps = create_all_lgn_layer_overlapping_regions(all_lgn_layers, params)
    adjusted_overlap_maps = mk_adjusted_overlapping_regions_wts(all_overlap_maps)

    return adjusted_overlap_maps


def create_v1_lif_network(
        params: do.SimulationParams
        ) -> do.LIFNetwork:

    # At some point, multiple sets of inputs may be simulated simultaneously
    # ... but for now, one at a time.
    v1_model = lif_model.mk_lif_v1(
        n_inputs=params.lgn_params.n_cells,
        lif_params=params.lif_params,
        n_trials=params.n_trials
        )

    return v1_model


def create_multi_v1_lif_network(
        params: do.SimulationParams,
        n_simulations: Optional[int] = None,
        overlap_map: Optional[Tuple[sfo.LGNOverlapMap, ...]] = None,
        ) -> do.LIFMultiNetwork:

    """

    n_simulations: for covering when partitions of full simulatinos are being run
    overlap_map: if provided, v1 network will be prepared to receive inputs from a variable
    number of inputs, as can be the case in if the overlapping regions are providing input to
    each V1 cell.  Generally, this is not effective for actual synchrony.
    """

    # At some point, multiple sets of inputs may be simulated simultaneously
    # ... but for now, one at a time.

    n_simulations_arg = (
            params.n_simulations
                if n_simulations is None else
            n_simulations
        )

    if overlap_map is not None:
        # IE, for each layer, number of overlap regions in that layer
        # each region will become a separate input/LGNCell/Synapse
        n_inputs_arg = tuple(len(layer) for layer in overlap_map)
        if not (len(n_inputs_arg) == params.n_simulations):
            raise exc.SimulationError(
                f'overlapping regions map should contain'
                )
        n_cells_arg = params.lgn_params.n_cells

        v1_model = lif_model.mk_multi_lif_v1(  # doing here to help typing know what input args are
            n_inputs=n_inputs_arg, n_cells= n_cells_arg,
            lif_params=params.lif_params,
            n_trials=params.n_trials,
            n_simulations=n_simulations_arg
            )
    else:
        n_inputs_arg = params.lgn_params.n_cells
        n_cells_arg = None

        v1_model = lif_model.mk_multi_lif_v1(  # duplicating to help typing know what input args are
            n_inputs=n_inputs_arg, n_cells= n_cells_arg,
            lif_params=params.lif_params,
            n_trials=params.n_trials,
            n_simulations=n_simulations_arg
            )


    return v1_model


# loop through LGN cells
# ... -> maybe here insert the additional trials

def mk_lgn_cell_response_analytically(
        cell: do.LGNCell,
        params: do.SimulationParams,
        actual_max_f1_amps: all_max_f1.F1MaxAmpDict,
        st_params: do.SpaceTimeParams,
        stim_params: do.GratingStimulusParams,
        temp_coords: Time[np.ndarray],
        rectified: bool = True
        ) -> do.ConvolutionResponse:

    spatial_product = srf.mk_spat_filt_temp_response(
        st_params, stim_params, cell.spat_filt, cell.location, cell.orientation,
        temp_coords)

    # temporal filter array
    tc = ff.mk_temp_coords(
        params.space_time_params.temp_res,
        tau=cell.temp_filt.parameters.arguments.tau
        )
    temp_filt = ff.mk_tq_tf(tc, cell.temp_filt)


    actual_max_f1_amp = all_max_f1.get_cell_actual_max_f1_amp(cell, actual_max_f1_amps)


    # cell_resp = convolve.mk_single_sf_tf_response(
    #         params.space_time_params, cell.spat_filt, cell.temp_filt,
    #         spat_filt, temp_filt,
    #         stim_params, stim_slice,
    #         filter_actual_max_f1_amp=actual_max_f1_amp.value,
    #         target_max_f1_amp=cell.max_f1_amplitude
    #         )

    resp: np.ndarray = convolve.convolve(spatial_product, temp_filt)[:temp_coords.value.shape[0]]

    # # adjustment parameters
    # for going from F1 SF and TF to convolution to accurate sinusoidal response
    adj_params = correction.mk_conv_resp_adjustment_params(
        st_params, stim_params, cell.spat_filt, cell.temp_filt,
        filter_actual_max_f1_amp=actual_max_f1_amp.value,
        target_max_f1_amp=cell.max_f1_amplitude
        )

    # # apply adjustment
    true_resp = correction.adjust_conv_resp(resp, adj_params)

    # # recification
    if rectified:
        true_resp[true_resp < 0] = 0

    return do.ConvolutionResponse(response=true_resp, adjustment_params=adj_params)


def mk_lgn_cell_response(
        cell: do.LGNCell,
        params: do.SimulationParams,
        stim_array: np.ndarray,
        actual_max_f1_amps: all_max_f1.F1MaxAmpDict,
        stim_params: do.GratingStimulusParams,
        ) -> do.ConvolutionResponse:


    # spatial filter array
    xc, yc = ff.mk_spat_coords(
                params.space_time_params.spat_res,
                sd=cell.spat_filt.parameters.max_sd()
                )

    spat_filt = ff.mk_dog_sf(
        x_coords=xc, y_coords=yc,
        dog_args=cell.oriented_spat_filt_params  # use oriented params
        )
    # Rotate array
    spat_filt = ff.mk_oriented_sf(spat_filt, cell.orientation)


    # temporal filter array
    tc = ff.mk_temp_coords(
        params.space_time_params.temp_res,
        tau=cell.temp_filt.parameters.arguments.tau
        )
    temp_filt = ff.mk_tq_tf(tc, cell.temp_filt)

    # slice stimulus
    spat_slice_idxs = stimulus.mk_rf_stim_spatial_slice_idxs(
        params.space_time_params, cell.spat_filt, cell.location)
    stim_slice = stimulus.mk_stimulus_slice_array(
        params.space_time_params, stim_array, spat_slice_idxs)

    # convolve
    actual_max_f1_amp = all_max_f1.get_cell_actual_max_f1_amp(cell, actual_max_f1_amps)
    cell_resp = convolve.mk_single_sf_tf_response(
            params.space_time_params, cell.spat_filt, cell.temp_filt,
            spat_filt, temp_filt,
            stim_params, stim_slice,
            filter_actual_max_f1_amp=actual_max_f1_amp.value,
            target_max_f1_amp=cell.max_f1_amplitude
            )

    return cell_resp

def loop_lgn_cells_mk_response(
        lgn: do.LGNLayer,
        params: do.SimulationParams,
        stim_array: Optional[np.ndarray],
        actual_max_f1_amps: all_max_f1.F1MaxAmpDict,
        stim_params: do.GratingStimulusParams,
        analytical: bool = False
        ) -> Tuple[do.ConvolutionResponse]:

    responses: Sequence[do.ConvolutionResponse] = []

    # ##### Loop through LGN cells
    if analytical:
        temp_cords = ff.mk_temp_coords(
            params.space_time_params.temp_res, params.space_time_params.temp_ext)

        for cell in lgn.cells:

            response = mk_lgn_cell_response_analytically(
                    cell, params, actual_max_f1_amps,
                    params.space_time_params, stim_params,
                    temp_coords=temp_cords
                )
            responses.append(response)
    else:
        if stim_array is None:
            raise ValueError('Stim array must be an array if not using analytical convolution')
        for cell in lgn.cells:
            # spatial filter array

            response = mk_lgn_cell_response(
                    cell,
                    params, stim_array, actual_max_f1_amps, stim_params
                )
            responses.append(response)

    responses = tuple(responses)
    return responses

# loop through N simulations
def loop_n_simulations(
        params: do.SimulationParams,
        all_lgn_layers: do.ContrastLgnLayerCollection,
        stim_array: np.ndarray,
        actual_max_f1_amps: all_max_f1.F1MaxAmpDict,
        stim_params: do.GratingStimulusParams,
        v1_model: do.LIFNetwork,
        stimulus_results_key: Optional[str] = None
        ) -> Tuple[do.SimulationResult, ...]:

    sim_results = []

    for sim_idx in range(params.n_simulations):

        print(f'Simulation {sim_idx}/{params.n_simulations} ({sim_idx/params.n_simulations:.2%})')

        # #### LGN Layer

        # Pull from cache of all LGN layers
        # depends on contrast of this stimulus and simulation number
        lgn = all_lgn_layers[stim_params.contrast][sim_idx]

        # lgn = cells.mk_lgn_layer(
        #     params.lgn_params,
        #     spat_res=params.space_time_params.spat_res,
        #     contrast=stim_params.contrast)

        # #### Create Spatial and Temporal Filter arrays
        # spat_filts: Sequence[np.ndarray] = []
        # temp_filts: Sequence[np.ndarray] = []

        responses = loop_lgn_cells_mk_response(
                lgn,
                params, stim_array, actual_max_f1_amps, stim_params,
                analytical=params.analytical_convolution
            )

        # be paranoid and use tuples ... ?
        # spat_filts, temp_filts, responses = (
        #     tuple(spat_filts), tuple(temp_filts), tuple(responses)
        #     )
        # #### Poisson spikes for all cells
        # Sigh ... the array is stored along with the adjustment params in an object
        # ... and they're all called "response(s)"
        response_arrays = tuple(
                response.response for response in responses
            )
        lgn_layer_responses = convolve.mk_lgn_response_spikes(
                params.space_time_params, response_arrays,
                n_trials = params.n_trials
            )

        # ### Simulate V1 Reponse
        spike_idxs, spike_times = (
            lif_model
            .mk_input_spike_indexed_arrays(lgn_response=lgn_layer_responses)
            )

        v1_model.reset_spikes(spike_idxs, spike_times)

        v1_model.run(params.space_time_params)

        v1_spikes = tuple(v1_model.spike_monitor.spike_trains().values())
        # shape: n_trials x time steps
        v1_mem_pot: np.ndarray = v1_model.membrane_monitor.v

        result = do.SimulationResult(
                stimulus_results_key = stimulus_results_key,
                n_simulation = sim_idx,
                spikes = v1_spikes,
                membrane_potential = v1_mem_pot,
                lgn_responses = lgn_layer_responses,
                n_trials = params.n_trials
            )

        sim_results.append(result)

    sim_results = tuple(sim_results)
    return sim_results


# looping through each stimulus
# def loop_stim_params(
#         all_stim_params: Tuple[do.GratingStimulusParams],
#         ):

def run_single_stim_simulation(
        params: do.SimulationParams,
        stim_params: do.GratingStimulusParams,
        all_lgn_layers: do.ContrastLgnLayerCollection,
        actual_max_f1_amps: all_max_f1.F1MaxAmpDict,
        v1_model: do.LIFNetwork
        ) -> Tuple[do.SimulationResult]:

    print(dedent(f'''\
        {stimulus.mk_stim_signature(params.space_time_params, stim_params)}
        '''))

    stimulus_results_key = f'{stimulus.mk_stim_signature(params.space_time_params, stim_params)}'

    # ### Load Stimulus
    stim_array = stimulus.load_stimulus_from_params(params.space_time_params, stim_params)

    # ### Actual Max F1 amplitudes

    # Do here, at the "top" so that only have to do once for a simulation
    # ... but I can't be bothered storing in disk and managing that.

    # ### Loop through N sims

    sim_results = loop_n_simulations(
            params, all_lgn_layers,
            stim_array, actual_max_f1_amps,
            stim_params, v1_model, stimulus_results_key
        )

    return sim_results


def run_simulation(
        params: do.SimulationParams,
        force_central_rf_locations: bool = False
        ) -> do.SimulationResults:

    # ## create stimulus

    all_stim_params = create_stimulus(params, force_central_rf_locations)

    # ## Create all LGN layers
    # Do this ahead of time so that exactly the same layer can be stimulated with all stimuli
    # if stimuli vary by contrast, there will be different layers for each contrast ...
    # (probably best to just use one contrast at a time)
    # The number of LGN layers is controlled by the `params.n_simulations`

    all_lgn_layers = create_all_lgn_layers(params, force_central_rf_locations)

    # ## Create V1 LIF network
    # Should be the same network just with new inputs each time
    v1_model = create_v1_lif_network(params)

    # ## CRUDE! Final results object
    # CRUDE but will probably get updated to be what I need
    # For now: {stim_signature: [v1.spikes]}
    results: Dict[str, Tuple[do.SimulationResult, ...]] = {}


    # ## Loop through each stim_params

    for stim_idx, stim_params in enumerate(all_stim_params):

        print(dedent(f'''\
            STIMULUS {stim_idx}/{len(all_stim_params)}:
            {stimulus.mk_stim_signature(params.space_time_params, stim_params)}
            '''))


        stimulus_results_key = f'{stimulus.mk_stim_signature(params.space_time_params, stim_params)}'

        # ### Load Stimulus
        stim_array = stimulus.load_stimulus_from_params(params.space_time_params, stim_params)

        # ### Actual Max F1 amplitudes

        # Do here, at the "top" so that only have to do once for a simulation
        # ... but I can't be bothered storing in disk and managing that.

        #
        actual_max_f1_amps = all_max_f1.mk_actual_max_f1_amps(stim_params=stim_params)


        # ### Loop through N sims

        sim_results = loop_n_simulations(
                params, all_lgn_layers,
                stim_array, actual_max_f1_amps,
                stim_params, v1_model, stimulus_results_key
            )

        results[stimulus_results_key] = sim_results


    # ## return results

    simulation_results = do.SimulationResults(
        params=params,
        lgn_layers=all_lgn_layers,
        results=results
        )
    return simulation_results


# # All in one multi simulation code

def mk_n_simulation_partitions(
        n_sims: int,
        max_sims: int = 500
        ) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
    n_partitions = math.ceil(n_sims / max_sims)

    partitioned_n_sims = [0]
    running_total_sims = n_sims
    for _ in range(n_partitions):
        if running_total_sims > max_sims:
            partitioned_n_sims.append(max_sims)
            running_total_sims -= max_sims
        elif running_total_sims == 0:
            break
        else:
            partitioned_n_sims.append(running_total_sims)

    partitioned_n_sims = tuple(partitioned_n_sims)

    partition_incs = list(itertools.accumulate(partitioned_n_sims))

    partition_idxs = tuple(
        (start, end)
            for start, end
            in zip(partition_incs[:-1], partition_incs[1:])
        )

    if len(partition_idxs) != n_partitions:
        raise ValueError(f'Incorrect number of partition indices generated (n: {n_partitions}, n_idxs: {len(partition_idxs)})')

    return n_partitions, partition_idxs


# # Main single stim simulation function
def run_single_stim_multi_layer_simulation(
        params: do.SimulationParams,
        stim_params: do.GratingStimulusParams,
        synch_params: do.SynchronyParams,
        lgn_layers: Union[Tuple[do.LGNLayer], do.ContrastLgnLayerCollection],
        lgn_overlap_maps: Optional[Tuple[sfo.LGNOverlapMap, ...]],
        log_print: bool = False, log_info: Optional[str] = None,
        save_membrane_data: bool = False
        ) -> Tuple[do.SimulationResult]:
    """

    """
    if isinstance(lgn_layers, dict):  # ContrastLgnLayerCollection is a dict with contrast val keys
        lgn_layers = lgn_layers[stim_params.contrast]
    n_lgn_layers = len(lgn_layers)
    # n_lgn_layers: number of simulations to run in this function, which may be less than the total
    #     nunmber of simulations for the whole simulation, due to compute resource limits.
    #     This number must correspond to the length of `all_lgn_layers`.
    #     If there is partitioning being done, it is up to the caller to manage the splitting
    #     of lgn_layers.

    # do once per simulation
    actual_max_f1_amps = all_max_f1.mk_actual_max_f1_amps(stim_params=stim_params)

    # ## Make stimulus (if necessary)
    if params.analytical_convolution:
        stim_array = None
    else:
        stim_array = stimulus.load_stimulus_from_params(params.space_time_params, stim_params)

    # ## Make v1 model

    # whether using synchrony or not, the v1 network is the same
    # the same number of inputs go to the same number of v1 cells
    # (as synchrony runs by duplicating synchronous spikes to "true LGN cells").

    v1_model = create_multi_v1_lif_network(
                params,
                # as partition of all simulations, use n_lgn_layers as proxy for size of partition
                n_simulations = n_lgn_layers
            )

    # ## LGN Responses

    # ### Convolution responses

    if log_print:
        def log_func1(i, log_info):
            if (i%20) == 0:
                print(f'{log_info} ... response for lgn layer {i}')
        log_func=log_func1
    else:
        def log_func2(i, _): pass
        log_func=log_func2


    # Nested tuple:    |- responses of cells for a single layer
    #                  V       V - tuple of responses for all layers
    #               ( (. . .), (. . . ) )
    # IE ... Nested iterable:
    #           inner = response of each cell of a layer
    #           outer = each layer
    all_responses: List[Tuple[do.ConvolutionResponse,...]] = list()
    for i, lgn in enumerate(lgn_layers):
        log_func(i, log_info)
        lgn_resp = loop_lgn_cells_mk_response(
                        lgn,
                        params, stim_array, actual_max_f1_amps, stim_params,
                        analytical=params.analytical_convolution
                    )
        all_responses.append(lgn_resp)


    # ### Flatten response arrays

    # #### Synchrony region response arrays ... create in flattening process

    # If synchrony, create synchronous overlapping regions response arrays
    if synch_params.lgn_has_synchrony:
        if lgn_overlap_maps is None:
            raise ValueError(f'lgn overlap maps must be provided if synchrony is to be implemented')

        response_arrays_collector: Sequence[np.ndarray] = list()

        # get temporal dimension size
        temp_array_size = all_responses[0][0].response.size
        # for each layer
        for i, layer_responses in enumerate(all_responses):
            overlap_map = lgn_overlap_maps[i]
            # n_regions = len(overlap_map)

            # put all original cell response arrays into a 2D array for better processing
            layer_responses_array = np.zeros(shape=(len(layer_responses), temp_array_size))
            for i, cell_response in enumerate(layer_responses):
                layer_responses_array[i] = cell_response.response

            # construct new response array based on number of overlapping regions
            # all_new_resps = np.zeros(shape=(n_regions, temp_array_size))

            # multiply out new responses
            # For each overlapping region ...
            for i, (cell_idxs, overlap_wts) in enumerate(overlap_map.items()):
                # cell_idxs: indices of lgn_layer.cells of the cells overlapping in this region
                # overlap_wts: contributions of each cell to this region, relative to cell's magnitude

                # multiply each cell's response by its weighting for the overlapping region
                # then sum over all the cells to create a single response array
                # 1: aggregate into single response (axis=0 means over the cells)
                # 2: get response arrays of all overlapping cells for this region
                # 3: multiply response arrays by wts
                #       - convert values to array (via tuple) which will be in same order as cell_idxs
                #       - slice with newaxis so that the values will be broadcast onto the cell resonses
                overlapping_region_response: np.ndarray = np.sum(           # 1
                    layer_responses_array[cell_idxs, :]                     # 2
                    *
                    np.array(tuple(overlap_wts.values()))[:, np.newaxis]    # 3
                    ,
                    axis=0
                    )

                # add response array to giant flattened list of response arrays
                response_arrays_collector.append(overlapping_region_response)

        # Flattened tuple of response arrays: (r11, r12, ... r1n, ... r21, r22, ... r_mn)
        #   where m = number of layers, n = number of OVERLAPPING REGIONS per layer
        #     /- Temporal response of a single OVERLAPPING REGION
        #     |                   /- Response arrays of each REGION from a single LGN layer
        #     |                   V     of length `len(overlap_map)` FOR EACH LAYER
        #     V               |--------------------|
        #  ( array[], array[], ... array[], array[] )
        response_arrays = tuple(response_arrays_collector)


    # #### No Synchrony ... just flattening

    else:
        # Flattened tuple of response arrays: (r11, r12, ... r1n, ... r21, r22, ... r_mn)
        #   where m = number of layers, n = number of cells per layer
        #     |- Temporal response of a single cell
        #     |                   |- Response arrays of each cell from a single LGN layer
        #     |                   V     of length lgn_params.n_cells (ie, `n` input cells)
        #     V               |--------------------|
        #  ( array[], array[], ... array[], array[] )
        response_arrays = tuple(
                single_response.response
                for responses in all_responses  # first layer is responses for a single layer
                    for single_response in responses # second layer is cellular responses within a layer
            )


    # spikes

    if log_print:
        print(f'{log_info} ... spikes for lgn responses')

    # adapt function to react to when n_sims is passed in
    # should then parse response arrays as being not just (cell1, cell2, ... celln)
    # but (cell11, cell12, ... cell21, cell22, ... cell_mn)

    # Where are trials?
    # They are only relevant for poisson spikes, so they arise in the data structures here

    # produce number of inputs per layer if synchrony is being used
    if synch_params.lgn_has_synchrony and (lgn_overlap_maps is not None):
        n_inputs = [len(overlap_map) for overlap_map in lgn_overlap_maps]
    else:
        n_inputs = params.lgn_params.n_cells

    # response objects flattened in order of (trials x lgn_layers)
    # eg (trial1-layer1, trial2-layer1, trial1-layer2, trial2-layer2, ...)
    lgn_layer_responses = convolve.mk_lgn_response_spikes(
            params.space_time_params,
            response_arrays = response_arrays,
            n_trials = params.n_trials,
            n_lgn_layers=n_lgn_layers,
            n_inputs=n_inputs  # either int or list of ints if synchrony and variable n cells per layer
        )

    # v1 model run and collect results

    if log_print:
        print(f'{log_info} ... running V1 model')

    # Should be in same order as above
    # but just flattened arrays of spike_idxs ... ie, spike_idx refers to which
    # response object in lgn_layer_responses

    if synch_params.lgn_has_synchrony:
        if lgn_overlap_maps is None:
            raise ValueError(f'lgn overlap maps must be provided if synchrony is to be implemented')

        spike_idxs, spike_times, all_spike_times = (
            lif_model
            .mk_input_spike_indexed_arrays(
                lgn_response=lgn_layer_responses,
                overlapping_region_map = lgn_overlap_maps,
                synchrony_params = synch_params,
                temp_ext = params.space_time_params.temp_ext,
                n_layers = n_lgn_layers,
                n_trials = params.n_trials,
                n_cells = params.lgn_params.n_cells
                )
            )

        # pull out newly collated synchronous lgn responses

        start_idxs, end_idxs = (
            (start_idxs := np.arange(0, len(all_spike_times), params.lgn_params.n_cells)),
            (start_idxs + 30)
            )

        # check
        if (
                    # final idx should be total len of spike_times sequence (ie n cells in all trial_layers)
                    (end_idxs[-1] != (n_lgn_layers * params.n_trials * params.lgn_params.n_cells))
                    or
                    # number of indices should be same as number of trial_layers (ie len of lgn_response)
                    (len(start_idxs) != len(lgn_layer_responses))
                    or
                    # number of indices should be same as number of trial_layers
                    (len(lgn_layer_responses) != (n_lgn_layers * params.n_trials))
                ):

            raise exc.SimulationError("Trial, layer, cell indices don't match expected sizes")

        all_spike_times = tuple(all_spike_times)

        new_lgn_layer_spike_times = tuple(  # trial_layer_responses
            # all spike times for each cell in trial_layer
            tuple(all_spike_times[start_idx:end_idx])
            for start_idx, end_idx in zip(start_idxs, end_idxs)
            )

        #######
        # How back into lgn response objects!?

        # each tuple in new_lgn_layer_spike_times should correspond to each LGNLayerResponse
        # ... in lgn_layer_responses
        # so simply loop through and replace the lgn_response.cell_spike_times tuple
        # ... with the tuples in new_lgn_layer_spike_times
        #######

        for i, lgn_layer_response in enumerate(lgn_layer_responses):
            # MUTATE!
            lgn_layer_response.cell_spike_times = new_lgn_layer_spike_times[i]



    else:
        spike_idxs, spike_times, _ = (
            lif_model
            .mk_input_spike_indexed_arrays(lgn_response=lgn_layer_responses )
            )

    v1_model.reset_spikes(spike_idxs, spike_times, spikes_sorted=False)
    v1_model.run(params.space_time_params)

    # return results

    # gotta reshape / organise results
    #

    # (trials x lgn_layers)
    v1_spikes = tuple(v1_model.spike_monitor.spike_trains().values())
    # shape: (n_trials x n_lgn_layers) x time steps
    v1_mem_pot: np.ndarray = v1_model.membrane_monitor.v

    # One simulation result for each layer (including trials)

    stimulus_results_key = f'{stimulus.mk_stim_signature(params.space_time_params, stim_params)}'

    results: List[do.SimulationResult] = []
    for n_lgn_layer in range(n_lgn_layers):
        spikes = tuple(
                v1_spikes[
                    0 + (n_lgn_layer*params.n_trials) : params.n_trials + (n_lgn_layer*params.n_trials)
                ]
            )

        if save_membrane_data:
            membrane_potential = v1_mem_pot[
                0 + (n_lgn_layer * params.n_trials) : params.n_trials + (n_lgn_layer * params.n_trials)
                ,
                :
            ]
        else:
            membrane_potential = None

        lgn_responses = lgn_layer_responses[
            0 + (n_lgn_layer*params.n_trials) : params.n_trials + (n_lgn_layer*params.n_trials)
        ]

        results.append(
            do.SimulationResult(
                stimulus_results_key=stimulus_results_key,
                n_simulation=n_lgn_layer,
                spikes=spikes,
                membrane_potential=membrane_potential,
                lgn_responses=lgn_responses,
                n_trials=params.n_trials
                )
            )

    results_tuple = tuple(results)

    return results_tuple



def _save_pickle_file(
        file: Path, obj: Any, overwrite: bool = False):

    if file.exists() and (not overwrite):
        raise FileExistsError(f'File already exists and no overwrite ({file})')

    # always add .pkl extension just in case it's missing
    with open(file.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(obj, f)


def _load_pickle_file(file: Path):

    if not file.exists():
        raise ValueError(f'File {file} does not exist')

    # always add .pkl extension just in case it's missing
    with open(file.with_suffix('.pkl'), 'rb') as f:
        obj = pickle.load(f)

    return obj



exp_dir_prefix = "exp_no_"


def mk_all_exp_dir(results_dir: Path):

    all_exp_dir = results_dir.glob(f'{exp_dir_prefix}*')

    return all_exp_dir


def save_simulation_results(
        results_dir: Path,
        sim_results: do.SimulationResults,
        comments: str = ''
        ):

    # move out into a separate function
    if not results_dir.exists():
        raise ValueError(f'Results directory does not exist: {results_dir}')

    # experiment folder
    all_exp_dirs = [
        p
            for p in mk_all_exp_dir(results_dir)
            # for p in results_dir.glob(f'{exp_dir_prefix}*')
            if p.is_dir()
        ]

    new_exp_dir_name = f'{exp_dir_prefix}{len(all_exp_dirs)}'
    new_exp_dir = results_dir / new_exp_dir_name
    if new_exp_dir.exists():
        raise FileExistsError(
            f'New experiment dir ({new_exp_dir}) already exists ... numbering awry?')
    new_exp_dir.mkdir()

    try:
        # meta data
        # probably only ever just the date/time and comments
        meta_data = {
            'creation_time': dt.datetime.utcnow().isoformat(),
            'comments': comments
        }
        meta_data_file = new_exp_dir/'meta_data.pkl'
        _save_pickle_file(meta_data_file, meta_data)

        # params
        simulation_params_file = new_exp_dir/'simulation_params.pkl'
        _save_pickle_file(simulation_params_file, sim_results.params)

        # lgn layer data
        lgn_layer_collection_file = new_exp_dir/'lgn_layers.pkl'

        # using an LGN record that reduces the size on disk by storing only a reference/key
        # to each RF rather than the whole thing as there are only a finite set and they are
        # already stored separately
        lgn_layer_collection_record = (
            cells.mk_contrast_lgn_layer_collection_record(sim_results.lgn_layers)
            )

        _save_pickle_file(lgn_layer_collection_file, lgn_layer_collection_record)

        # Results data
        results_data_file = new_exp_dir/'results_data.pkl'
        _save_pickle_file(results_data_file, sim_results.results)

    except Exception as e:
        # delete whole folder so that no bad results floating around
        shutil.rmtree(new_exp_dir)
        raise exc.SimulationError('Failed to save results') from e


def prep_results_dir(results_dir: Path, suppress_exc: bool = False):
    if results_dir.exists():
        message = f'Results directory already exists ... start new experiment'
        if not suppress_exc:
            raise ValueError(message)
        else:
            print(f'WARNING: {message}')

    else:
        results_dir.mkdir()


def prep_temp_results_dirs(results_dir: Path, n_stims: Union[int, Sequence]):

    if not isinstance(n_stims, int):
        n_stims = len(n_stims)

    if (not results_dir.exists()):
        raise ValueError(f'Results directory does not exist: {results_dir}')

    for stim in range(n_stims):
        stim_dir = results_dir / f'{stim:0>3}'
        if stim_dir.exists():
            raise ValueError(f'Stim dir ({stim_dir}) already exists ... results already organised')
        stim_dir.mkdir()


# # Main RUN for single stim for MP function
def run_partitioned_single_stim(
        params: do.SimulationParams,
        n_stim: int,
        stim_params: do.GratingStimulusParams,
        synch_params: do.SynchronyParams,
        lgn_layers: do.ContrastLgnLayerCollection,
        lgn_overlap_maps: Optional[Dict[do.ContrastValue, Tuple[sfo.LGNOverlapMap, ...]]],
        results_dir: Path,
        partitioned_sim_lgn_idxs: Tuple[Tuple[int, int], ...],
        log_print: bool = False,
        save_membrane_data: bool = False
        ):

    for n_partition, (start, end) in enumerate(partitioned_sim_lgn_idxs):
        if log_print:
            log_info = f'Stim: {n_stim}, part: {n_partition}'
            print(f'Running ... {log_info} ({dt.datetime.utcnow().isoformat()})')
        else:
            log_info = None


        partitioned_lgn_layer = lgn_layers[stim_params.contrast][start : end]
        contrast_specific_overlap_maps = (
                # pass in only the right overlap regions
                lgn_overlap_maps[stim_params.contrast][start : end]
                    if lgn_overlap_maps else
                None
            )

        partitioned_results = run_single_stim_multi_layer_simulation(
                params, stim_params,
                synch_params = synch_params,
                lgn_layers = partitioned_lgn_layer,
                lgn_overlap_maps = contrast_specific_overlap_maps,
                log_print=log_print, log_info=log_info,
                save_membrane_data=save_membrane_data
            )

        if log_print:
            print(f'{log_info} ... saving partitioned single stim results ({dt.datetime.utcnow().isoformat()})')

        save_single_stim_results(
            results_dir, partitioned_results,
            stim_n=n_stim, partition=n_partition
            )

    if log_print:
        print(f'Stim: {n_stim} ... saving all single stim results ({dt.datetime.utcnow().isoformat()})')

    save_merge_single_stim_results(results_dir, stim_n=n_stim)


def _mk_stim_results_dir(results_dir: Path, stim_n: int) -> Path:

    if (not results_dir.exists()):
        raise ValueError(f'Results directory does not exist: {results_dir}')

    stim_results_dir = results_dir / f'{stim_n:0>3}'

    return stim_results_dir


def _parse_stim_results_path(stim_results_file: Path) -> Optional[int]:

    pattern = re.compile(r'(\d\d\d).pkl')

    path_match = pattern.fullmatch(stim_results_file.name)

    if path_match is None:
        return None

    stim_n = int(path_match.group(1))

    return stim_n


def save_single_stim_results(
        results_dir: Path,
        results: Tuple[do.SimulationResult],
        stim_n: int,
        partition: int
        ):
    """

    stim_n: which of the multi_stim combos this simulation belongs to
    partition: which part of the full simulation
    """

    stim_results_dir = _mk_stim_results_dir(results_dir, stim_n)
    part_results_file_path = stim_results_dir / f'{partition:0>3}.pkl'
    if (not results_dir.exists()):
        raise ValueError(f'Results directory does not exist: {results_dir}')
    elif (not stim_results_dir.exists()):
        raise ValueError(f'Stim results directory does not exist: {stim_results_dir}')

    if part_results_file_path.exists():
        raise ValueError(f'Partition results file already exists: {stim_results_dir}')

    _save_pickle_file(part_results_file_path, results)


def save_merge_single_stim_results(
        results_dir: Path,
        stim_n: int
        ):

    stim_results_dir = _mk_stim_results_dir(results_dir, stim_n)
    if (not results_dir.exists()):
        raise ValueError(f'Results directory does not exist: {results_dir}')
    elif (not stim_results_dir.exists()):
        raise ValueError(f'Stim results directory does not exist: {stim_results_dir}')

    partitioned_results = []
    partitioned_results_files = stim_results_dir.glob('*.pkl')

    for result_file in partitioned_results_files:
        results = _load_pickle_file(result_file)
        partitioned_results.append(results)

    full_results = tuple(
            result
            for partial_results in partitioned_results
                for result in partial_results
        )

    _save_pickle_file(stim_results_dir.with_suffix('.pkl'), full_results)


def get_all_experiment_single_stim_results_files(
        results_dir: Path,
        multi_stim_combos: Tuple
        ) -> Dict[int, Path]:

    if (not results_dir.exists()):
        raise ValueError(f'Results directory does not exist: {results_dir}')

    all_result_files = results_dir.glob('*.pkl')

    # match numbers to n_stims
    result_files_idxs = {}
    for result_file in all_result_files:
        i = _parse_stim_results_path(result_file)
        if i is not None:
            result_files_idxs[i] = result_file

    # checking that idxs match what would be expected by the number of stim combos
    if not (
            # exclude None as not match
            sorted(list(k for k in result_files_idxs.keys()))
            ==
            list(range(len(multi_stim_combos)))
            ):
        raise exc.SimulationError(
            f"Stim result numbering doesn't match len of multi_stim comb: {len(multi_stim_combos)}")

    result_files_idxs = cast(Dict[int, Path], result_files_idxs)

    return result_files_idxs

def save_merge_all_results(
        results_dir: Path,
        multi_stim_combos: Tuple[do.GratingStimulusParams],
        params: do.SimulationParams,
        ):

    result_files_idxs = get_all_experiment_single_stim_results_files(
        results_dir, multi_stim_combos)

    # loop through each n and stim: load file, key with stimulus_sig, save dict

    all_results = dict()

    for n_stim, stim_params in enumerate(multi_stim_combos):
        result_file = result_files_idxs[n_stim]
        results: Tuple[do.SimulationResult] = _load_pickle_file(result_file)

        stim_sig = stimulus.mk_stim_signature(params.space_time_params, stim_params)
        all_results[stim_sig] = results

    _save_pickle_file(results_dir / 'results_data.pkl', all_results)


def load_meta_data(exp_results_dir: Path) -> Dict:

    meta_data_file = exp_results_dir/'meta_data.pkl'
    meta_data = _load_pickle_file(meta_data_file)

    return meta_data


def load_sim_params(exp_results_dir: Path) -> do.SimulationParams:
    params_file = exp_results_dir/'simulation_params.pkl'
    params = _load_pickle_file(params_file)

    return params


def load_simulation_results(
        results_dir: Path,
        exp_dir: Path
        ) -> Tuple[dict, do.SimulationResults]:

    exp_results_dir = results_dir / exp_dir

    if not exp_results_dir.exists():
        raise ValueError(f'Directory {exp_results_dir} does not exist')

    # meta_data_file = exp_results_dir/'meta_data.pkl'
    # meta_data = _load_pickle_file(meta_data_file)
    meta_data = load_meta_data(exp_results_dir)

    # params
    # simulation_params_file = exp_results_dir/'simulation_params.pkl'
    # simulation_params = _load_pickle_file(simulation_params_file)
    simulation_params = load_sim_params(exp_results_dir)

    # lgn layer data
    lgn_layer_collection_file = exp_results_dir/'lgn_layers.pkl'

    # using an LGN record that reduces the size on disk by storing only a reference/key
    # to each RF rather than the whole thing as there are only a finite set and they are
    # already stored separately

    lgn_layer_collection_record = _load_pickle_file(lgn_layer_collection_file)
    lgn_layer_collection = (
        cells.mk_contrast_lgn_layer_collection_from_record(lgn_layer_collection_record)
        )

    # Results data
    results_data_file = exp_results_dir/'results_data.pkl'
    results_data = _load_pickle_file(results_data_file)

    sim_results = do.SimulationResults(
            params=simulation_params,
            lgn_layers = lgn_layer_collection,
            results = results_data
        )

    return meta_data, sim_results


