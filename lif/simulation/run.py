"""Create responses of a LGN layer to a stimulus

"""

from textwrap import dedent
from typing import Sequence, Dict, Tuple

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



def run_simulation(
    params: do.SimulationParams,
    force_central_rf_locations: bool = False):

    # ## create stimulus

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

    # ## Create all LGN layers
    # Do this ahead of time so that exactly the same layer can be stimulated with all stimuli
    # The number of LGN layers is controlled by the `params.n_simulations`

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

    # ### Delete all ori bias params
    # just take up unneeded space ... all needed for is to create ori biased versions
    # estimates (from pickling) layer goes from 911kb to 32kb in size when removing ori_bias_params
    # ACTUALLY ... this doesn't make sense because ... each repeat of the spat filt is a reference
    # ... to the same object (same object id) all the way in the filters module ...
    # ... so this breaks things (HARD) ... and is unnecessary
    # ... for saving on RAM.  Might be necessary to save on disk space for pickling though.
    # but ... how do without breaking things for the run time!!

    # print('!!! Deleting all ori_bias_params from all spatial filters ... no longer necessary for simulation!!!')
    # for _, layers in all_lgn_layers.items():
    #     for layer in layers:
    #         for cell in layer.cells:
    #             del cell.spat_filt.ori_bias_params


    # ## Create V1 LIF network
    # Should be the same network just with new inputs each time
    # At some point, multiple sets of inputs may be simulated simultaneously
    # ... but for now, one at a time.
    v1_model = lif_model.mk_lif_v1(
        n_inputs=params.lgn_params.n_cells,
        lif_params=params.lif_params
        )

    # ## CRUDE! Final results object
    # CRUDE but will probably get updated to be what I need
    # For now: {stim_signature: [v1.spikes]}
    results = {}


    # ## Loop through each stim_params


    for stim_idx, stim_params in enumerate(all_stim_params):

        print(dedent(f'''\
            STIMULUS {stim_idx}/{len(all_stim_params)}:
            {stimulus.mk_stim_signature(params.space_time_params, stim_params)}
            '''))


        stimulus_results_key = f'{stimulus.mk_stim_signature(params.space_time_params, stim_params)}'
        results[stimulus_results_key] = []

        # ### Load Stimulus
        stim_array = stimulus.load_stimulus_from_params(params.space_time_params, stim_params)

        # ### Actual Max F1 amplitudes

        # Do here, at the "top" so that only have to do once for a simulation
        # ... but I can't be bothered storing in disk and managing that.

        #
        actual_max_f1_amps = all_max_f1.mk_actual_max_f1_amps(stim_params=stim_params)


        # ### Loop through N sims

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
            spat_filts: Sequence[np.ndarray] = []
            temp_filts: Sequence[np.ndarray] = []
            responses: Sequence[do.ConvolutionResponse] = []

            # ##### Loop through LGN cells
            for cell in lgn.cells:
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

                spat_filts.append(spat_filt)

                # temporal filter array
                tc = ff.mk_temp_coords(
                    params.space_time_params.temp_res,
                    tau=cell.temp_filt.parameters.arguments.tau
                    )
                temp_filt = ff.mk_tq_tf(tc, cell.temp_filt)
                temp_filts.append(temp_filt)

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

                responses.append(cell_resp)

            # be paranoid and use tuples ... ?
            spat_filts, temp_filts, responses = (
                tuple(spat_filts), tuple(temp_filts), tuple(responses)
                )
            # #### Poisson spikes for all cells
            # Sigh ... the array is stored along with the adjustment params in an object
            # ... and they're all called "response(s)"
            response_arrays = tuple(
                    response.response for response in responses
                )
            lgn_layer_responses = convolve.mk_lgn_response_spikes(
                    params.space_time_params, response_arrays
                )

            # ### Simulate V1 Reponse
            spike_idxs, spike_times = (
                lif_model.
                mk_input_spike_indexed_arrays(lgn_response=lgn_layer_responses) )

            v1_model.reset_spikes(spike_idxs, spike_times)

            v1_model.run(params.space_time_params)

            results[stimulus_results_key].append(
                    {
                    'spikes': v1_model.spike_monitor.spike_trains()[0],
                    'membrane_potential': v1_model.membrane_monitor.v[0],
                    'lgn_responses': lgn_layer_responses,
                    'lgn_spikes': spike_times
                    }
                )


    # ## return results

    return results




