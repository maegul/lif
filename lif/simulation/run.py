"""Create responses of a LGN layer to a stimulus

"""

from ..utils import data_objects as do

from ..lgn import cells
from . import all_filter_actual_max_f1_amp as all_max_f1


# def mk_lgn_layer_responses(
#         lgn_params: do.LGNParams,
#         stim_params: do.GratingStimulusParams,
#         space_time_params: do.SpaceTimeParams
#         ):
#     """For LGN layer and stimulus parameters, return firing rate responses
#     """

#     lgn_layer = cells.mk_lgn_layer(lgn_params, space_time_params.spat_res)



def run_simulation(
        n: int,
        space_time_params: do.SpaceTimeParams,
        stim_params: do.GratingStimulusParams,
        lgn_params: do.LGNParams,
        v1_params: do.V1Params):


    actual_max_f1_amps = all_max_f1.mk_actual_max_f1_amps(stim_params=stim_params)

    # probably repeat n times (for each simulation) from here ... but for now, just write flat for 1

    lgn = cells.mk_lgn_layer(lgn_params, spat_res=space_time_params.spat_res)



    return lgn




