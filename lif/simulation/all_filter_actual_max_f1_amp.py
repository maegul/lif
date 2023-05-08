"""Create actual max f1 for all filter combinations and store as module attribute

All filter combinations are drawn from filters listed in the filter_index

Example:

    stim_params = do.GratingStimulusParams(
        spat_freq_x, temp_freq,
        orientation=orientation,
        amplitude=stim_amp, DC=stim_DC,
        contrast=do.ContrastValue(0.4)
    )
    actual_max_f1_amps = all_max_f1.mk_actual_max_f1_amps(stim_params=stim_params)

    # print results out
    for k, max_f1 in sorted(
            actual_max_f1_amps.items(), key=lambda x: x[1].value.max_amp
            ):
        print(
            f'{k[0] + " - " + k[1]:<30}',
            f'{max_f1.value.max_amp:<8.3f}',
            f'{max_f1.spat_freq.cpd:<5.3f}',
            f'{max_f1.temp_freq.hz:<5.3f}'
            )
"""

# # Imports
from typing import Optional, Dict, Tuple
from itertools import product

from ..utils import data_objects as do
from ..receptive_field.filters import filters
from ..convolution import correction


# # Get all filters and product

spatial_filters = filters.spatial_filters
"Filter index key : spatial filter"
temporal_filters = filters.temporal_filters
"Filter index key : temporal filter"



all_filter_combinations_keys = tuple(
        product(spatial_filters.keys(), temporal_filters.keys())
    )


# Find max for each combination

F1MaxAmpDict = Dict[Tuple[str, str], do.LGNActualF1AmpMax]

def mk_actual_max_f1_amps(
        stim_params: do.GratingStimulusParams,
        contrast_params: Optional[do.ContrastParams] = None
        ) -> F1MaxAmpDict:

    actual_max_f1_amps = {}

    for filter_keys in all_filter_combinations_keys:
        sf, tf = spatial_filters[filter_keys[0]], temporal_filters[filter_keys[1]]
        actual_max_amp = correction.mk_actual_filter_max_amp(
                sf, tf,
                stim_params.contrast, contrast_params=contrast_params
            )
        actual_max_f1_amps[filter_keys] = actual_max_amp

    return actual_max_f1_amps


def get_cell_actual_max_f1_amp(
        lgn_cell: do.LGNCell,
        all_max_f1_amps: F1MaxAmpDict
        ):

    cell_index_key = (
            filters.reverse_spatial_filters[lgn_cell.spat_filt.key],
            filters.reverse_temporal_filters[lgn_cell.temp_filt.key]
        )

    max_f1_amp = all_max_f1_amps[cell_index_key]

    return max_f1_amp
