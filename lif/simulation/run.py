"""Create responses of a LGN layer to a stimulus

"""

from textwrap import dedent
from typing import Any, Sequence, Dict, Tuple, Optional
from pathlib import Path
import shutil
import datetime as dt
import pickle

import numpy as np

from ..utils import data_objects as do, exceptions as exc

from ..receptive_field.filters import filter_functions as ff

from ..convolution import convolve
from ..lgn import cells
from ..stimulus import stimulus
from . import (
    all_filter_actual_max_f1_amp as all_max_f1,
    leaky_int_fire as lif_model
    )


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


do.ContrastLgnLayerCollection

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


# loop through LGN cells
# ... -> maybe here insert the additional trials
def mk_lgn_cell_response(
        cell: do.LGNCell,
        params: do.SimulationParams,
        stim_array: np.ndarray,
        actual_max_f1_amps: all_max_f1.F1MaxAmpDict,
        stim_params: do.GratingStimulusParams
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
        stim_array: np.ndarray,
        actual_max_f1_amps: all_max_f1.F1MaxAmpDict,
        stim_params: do.GratingStimulusParams
        ) -> Tuple[do.ConvolutionResponse]:

    responses: Sequence[do.ConvolutionResponse] = []

    # ##### Loop through LGN cells
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
                params, stim_array, actual_max_f1_amps, stim_params
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

def load_meta_data(exp_results_dir: Path) -> Dict:

    meta_data_file = exp_results_dir/'meta_data.pkl'
    meta_data = _load_pickle_file(meta_data_file)

    return meta_data

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
    simulation_params_file = exp_results_dir/'simulation_params.pkl'
    simulation_params = _load_pickle_file(simulation_params_file)

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


