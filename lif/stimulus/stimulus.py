from typing import Optional, Union, cast, Tuple, overload, Iterable, Dict, List
from textwrap import dedent
from dataclasses import replace
import re
from pathlib import Path
import itertools

import numpy as np

# import matplotlib.pyplot as plt

# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui

# import scipy.stats as st

# I'd prefer to move the coords functions to utls.coords.py
from ..receptive_field.filters import filter_functions as ff
from .. import lgn
from ..utils import settings, exceptions as exc
from ..utils.units.units import (
    SpatFrequency, TempFrequency, ArcLength, Time,
    val_gen, scalar)
from ..utils import data_objects as do

# from . import data_objects as do, estimate_real_amp_from_f1 as est_amp

PI: float = np.pi


def mk_sinstim(
        space_time_params: do.SpaceTimeParams,
        grating_stim_params: do.GratingStimulusParams,
        ) -> np.ndarray:
    '''Generate sinusoidal grating stimulus

    Returns 3D array: x, y, t

    Grating "lines" are oriented according to "orientation" in grating_stim_params
    Modulation or propogation over time is in "direction" in grating_stim_params

    Spatial and Temporal coords are generated according to space_time_params

    array_dtype, if provided, will alter the dtype of the array
    '''

    # creating coords
    xc, yc, tc = ff.mk_spat_temp_coords(
        spat_ext=space_time_params.spat_ext, temp_ext=space_time_params.temp_ext,
        spat_res=space_time_params.spat_res, temp_res=space_time_params.temp_res,
        array_dtype=space_time_params.array_dtype
        )

    # within the cos, the coords are in SI (base: degs, seconds) but the factors
    # are in angular frequency units, which makes cos work in SI units
    # (default one cycle per second or degree)
    img = (
        grating_stim_params.DC +
        (
            grating_stim_params.amplitude
            *
            np.cos(
                (grating_stim_params.spat_freq_x.cpd_w * xc.base)  # x
                +
                (grating_stim_params.spat_freq_y.cpd_w * yc.base)  # y
                # subtract temporal so that "wave" moves in positive direction
                -
                (grating_stim_params.temp_freq.w * tc.base)  # time
                )
            )
        )

    return img


@overload
def estimate_max_stimulus_spatial_ext_for_lgn(
        spat_res: ArcLength[int],
        lgn_layer: do.LGNParams,
        n_cells: Optional[int] = None,
        safety_margin_increment: float = 0,
        ) -> ArcLength[scalar]: ...
@overload
def estimate_max_stimulus_spatial_ext_for_lgn(
        spat_res: ArcLength[int],
        lgn_layer: do.LGNLayer,
        n_cells: None = None,
        safety_margin_increment: float = 0,
        ) -> ArcLength[scalar]: ...
def estimate_max_stimulus_spatial_ext_for_lgn(
        spat_res: ArcLength[int],
        lgn_layer: Union[do.LGNParams, do.LGNLayer],
        n_cells: Optional[int] = None,
        safety_margin_increment: float = 0,
        ) -> ArcLength[scalar]:
    """Spatial ext (width, not radial) to encompass the LGN layer given the params

    Spat extent returned is a width or diameter not a radius.
    Calculated by rendering a set of cell parameters (especially location),
    finding most distal location and greatest std dev of all the spat filters,
    adding them, then finding the largest such extent amongst all the cells.
    Important to note that this estimate **relies entirely** on the statistical
    generation of cell locations.

    Args:
        n_cells: If provided, the lgn params are adjusted to have the provided number
                    of cells.
                 Useful for dealing with the statistical nature of the estimate and so
                 use more cells to get a better estimate.
                 Must be provided along with an `LGNParams` object.
         safety_margin_increment: percentage (fraction) to add to estimate as a safety
                                  margin.  Estimate is multiplied by `(1 + safety_margin_increment)`.
    Examples:
        >>> spat_res = ArcLength(1, 'mnt')
        >>> lgn_params = lgn.demo_lgnparams
        >>> estimate_max_stimulus_spatial_ext_for_lgn(spat_res, lgn_params, n_cells=100)
        ArcLength(value=76, unit='mnt')
    """

    if isinstance(lgn_layer, do.LGNParams):
        # if n_cells argument provided
        if n_cells:
            lgn_layer = replace(lgn_layer, n_cells=n_cells)
        lgn_layer = lgn.mk_lgn_layer(lgn_layer, spat_res)

    # lists for largest distance and sd_extent, both will be put into base units (scalar)
    largest_locs: Iterable[scalar] = []
    largest_sd_extents: Iterable[scalar] = []

    # Go through each cell and get the largest distance (from x or y) and the largest sd extent
    for cell in lgn_layer.cells:
        # remain agnostic about the default orientation of the location elongation
        # ... and just take the largest location
        largest_loc = max(
            (cell.location.y, cell.location.x),
            # take base unit, and us abs as large negative loc just means down/left
            # ... with origin in the middle
            key = lambda a: abs(a.base)
            )

        largest_locs.append(abs(largest_loc.base))

        # most likely will get largest surround sd, so using oriented no so necessray
        # ... but using just in case central sd is large
        largest_sd = cell.oriented_spat_filt_params.max_sd()
        largest_sd_extent = (
            largest_sd.base * settings.simulation_params.spat_filt_sd_factor)

        largest_sd_extents.append(largest_sd_extent)

    # Get the largest distance of all cells and the largest sd extent of all cells
    largest_loc = max(largest_locs)
    largest_sd_extent = max(largest_sd_extents)
    # presume this "worst case scenario" and find the spatial width that this would require
    # round up to spatial res also
    max_furthest_coords = ff.round_coord_to_res(
        # 1. double to be a spatial extent not a radial
        # 2. absolute as negative only means down/left from origin
        # 3. already in base unit (scalar)
        # 4. already put into base unit (ie scalar)
        # 5. add safety increment by treating as a percentage (eg, 10% -> * 1.1)
        #         1     2     3              4                  5
        #         V     V     V              V                  V
        ArcLength(2 * (abs(largest_loc) + largest_sd_extent) * (1 + safety_margin_increment) ),
        spat_res, high=True
        )

    # find largest of all cells
    # max_furthest_coords = max(cells_furthest_canvas_coords, key=lambda a: a.base)

    return max_furthest_coords

# +
def mk_stim_signature(
    st_params: do.SpaceTimeParams,
    stim_params: do.GratingStimulusParams
    )->str:
    """Single string representing parameters for use as a file name


    """

    stparams_signature = []
    for field in st_params.__dataclass_fields__.keys():
        field_value = getattr(st_params, field)

        # is a unit, so get value and description out
        if hasattr(field_value, 'unit'):
            value = field_value.value
            description = field_value.unit
        # handle scalar or string
        elif isinstance(field_value, (int, float, str)):
            value = field_value
            description = type(field_value).__name__
        else:
            raise ValueError(f'Field cannot be parsed into string: {field, field_value}')
        signature_element = f'{field}={value}({description})'
        stparams_signature.append(signature_element)

    stimparams_signature = []
    for field in stim_params.__dataclass_fields__.keys():
        field_value = getattr(stim_params, field)
        # handle unit object
        if hasattr(field_value, 'unit'):
            value = field_value.value
            description = field_value.unit
        # handle scalar
        elif isinstance(field_value, (int, float)):
            value = field_value
            description = type(field_value).__name__
        # Handle others ...
        # handle contrast
        elif isinstance(field_value, do.ContrastValue):
            value = field_value.contrast
            description = 'contrast'
        else:
            raise ValueError(f'Field cannot be parsed into string: {field, field_value}')
        signature_element = f'{field}={value}({description})'
        stimparams_signature.append(signature_element)

    signature = f'STIMULUS|STP|{"|".join(stparams_signature)}|STIMP|{"|".join(stimparams_signature)}'

    return signature

# -
# +
def mk_params_from_stim_signature(
        signature:str
        )->Tuple[do.SpaceTimeParams, do.GratingStimulusParams]:
    """Provide data structures for generating a stimulus from a string signature

    This is ... not good code ... hacky and patched a few times to just work.
    If it starts to cause some trouble, consider re-working the whole idea ...
    ... maybe a json index with unique IDs and the actual JSON of the data_objects?
    """
    elements =signature.split('|')
    if not elements[:2] == ['STIMULUS', 'STP']:
        raise ValueError('signature is not a stimulus signature')

    param_pattern=re.compile(r'([\w_]+)=([\-\+]?[\d\.]+e?[\-\+\d]*|\w+)\((\w+)\)')
    int_value_pattern = re.compile(r'[\-\+]?\d+')  # for parsing ints as integers (no decimal!)
    str_value_pattern = re.compile(r'\w+')
    # for long floats with (e+6) exponentials
    # no need to parse normal floats (with decimal) as float is the fallback
    long_float_pattern = re.compile(r'[\d\.]+e[-+]?\d+')
    unit_type_look_up = {
        ('deg', 'mnt', 'sec', 'rad'): ArcLength,
        ('s', 'ms', 'us'): Time,
        ('cpd', 'cpm', 'cpd_w'): SpatFrequency,
        ('hz', 'w'): TempFrequency,
        ('int',): lambda v,_: int(v),  # proxy to maintain consistent interface
        ('float',): lambda v,_: float(v),
        ('str',): lambda v,_: str(v),
        ('contrast',): lambda v,_: do.ContrastValue(contrast=v)
    }

    signature_start_idx = elements.index('STIMULUS')
    space_time_start_idx = elements.index('STP')
    stim_params_start_idx = elements.index('STIMP')
    placeholder_idxs = (signature_start_idx, space_time_start_idx, stim_params_start_idx)

    st_params_dict = {}
    stim_params_dict = {}
    for idx, param in enumerate(elements):
        # skip place holders
        if idx in placeholder_idxs:
            continue

        param_parts = param_pattern.fullmatch(param)
        if not (param_parts):
            raise ValueError(f'Signature component does not parse: {param}')
        # the OR creates new groups (group 4 and 5 are now when there is no unit)
        # start_group = 1 if param_parts.group(1) else 4

        attribute = param_parts.group(1)
        value_match = param_parts.group(2)
        # determine basic qualities of the value for conversion
        is_int_type = (int_value_pattern.fullmatch(value_match) is not None)
        is_str_type = (str_value_pattern.fullmatch(value_match) is not None)
        long_float_value_match = long_float_pattern.fullmatch(value_match)

        if is_int_type:
            value = int(param_parts.group(2))
        elif is_str_type:
            value = str(param_parts.group(2))
        elif long_float_value_match:
            value = float(long_float_value_match.string)
        # float is default fallback
        else:
            value = float(param_parts.group(2))

        unit = param_parts.group(3)

        # find unit_type by simply iterating until found,
        # ... then use the loop variable directly after (bit hacky!)
        for possible_units, unit_type in unit_type_look_up.items():
            if unit in possible_units:
                break
        else:  # if unit not found in lookup
            raise ValueError(f'unit for param not found in lookup: {unit}')

        if space_time_start_idx < idx < stim_params_start_idx:
            st_params_dict[attribute] = unit_type(value, unit)
        else:
            stim_params_dict[attribute] = unit_type(value, unit)

    return (do.SpaceTimeParams(**st_params_dict), do.GratingStimulusParams(**stim_params_dict))
# -

# +
def save_stim_array_to_file(
        st_params: do.SpaceTimeParams,
        stim_params: do.GratingStimulusParams,
        stimulus: np.ndarray,
        overwrite: bool = False
        ):

    data_dir = settings.get_data_dir()
    file_name = Path(mk_stim_signature(st_params, stim_params) + '.npy')
    file_path = data_dir/file_name

    if file_path.exists() and (not overwrite):
        raise FileExistsError(f'Must set overwrite to True to overwrite: {file_name}')

    np.save(file_path, stimulus)
# -
# +
def mk_save_stim_array(
        st_params: do.SpaceTimeParams,
        stim_params: do.GratingStimulusParams,
        overwrite: bool = False
        ):
    save_stim_array_to_file(
        st_params, stim_params,
        mk_sinstim(st_params, stim_params),
        overwrite=overwrite
        )
# -

def load_stim_array_from_file(
        file_name: Path
        ) -> np.ndarray:

    data_dir = settings.get_data_dir()
    file_path = data_dir/file_name.with_suffix('.npy')

    if not file_path.exists():
        raise FileNotFoundError(f'Stimulus not found at {file_path}')

    stimulus_array = np.load(file_path)

    return stimulus_array


def load_stimulus_from_params(st_params, stim_params) -> np.ndarray:
    """Load from data directory based on params ... presumes already cached
    """

    signature = mk_stim_signature(st_params, stim_params)
    stim_file_name = Path(signature + '.npy')

    stimulus = load_stim_array_from_file(stim_file_name)

    return stimulus


def get_params_for_all_saved_stimuli(
        ) -> Dict[do.SpaceTimeParams, List[do.GratingStimulusParams]]:

    data_dir = settings.get_data_dir()
    all_stim_files = data_dir.glob('STIMULUS*')

    all_stim_params = (
        mk_params_from_stim_signature(file.stem)
            for file in all_stim_files
        )

    all_params = {}
    for st, group in itertools.groupby(all_stim_params, key=lambda x: x[0]):
        if st not in all_params:
            # use set for comparisons later as should all be unique anyway
            all_params[st] = set()
        all_params[st].update(set(g[1] for g in group))

    return all_params


def print_params_for_all_saved_stimuli():

    all_params = get_params_for_all_saved_stimuli()

    st_params = sorted(
            list(all_params.keys()),
            key=lambda t: (
                t.temp_res.ms,t.temp_ext.ms,t.spat_res.mnt,t.spat_ext.mnt,)
        )

    for st in st_params:
        spat_params = f'{st.spat_ext.value} {st.spat_ext.unit} / {st.spat_res.value} {st.spat_res.unit}'
        temp_params = f'{st.temp_ext.value} {st.temp_ext.unit} / {st.temp_res.value} {st.temp_res.unit}'
        print(f'\n{spat_params:>16} | {temp_params}')
        # sort for printing
        stim_params = sorted(
            all_params[st],
            # last items is used first for sorting
            key=lambda t: (
                t.amplitude, t.DC, t.temp_freq.hz, t.spat_freq.cpd, t.orientation.deg)
            )
        for stim in stim_params:
            ori = f'Ori: {stim.orientation.deg:<5}'
            freqs = f'SF: {stim.spat_freq.cpd:<5} | TF: {stim.temp_freq.hz:<5}'
            print(f'\t{ori} | {freqs} (Amp: {stim.amplitude:<2}, DC: {stim.DC:<3}, Cont: {stim.contrast.contrast})')



def mk_multi_stimulus_params(
        multi_params: do.MultiStimulusGeneratorParams
        ) -> Tuple[do.GratingStimulusParams]:
    """Make multiple stimulus params from all combinations of parameters provided

    Where default value is `[None]`, the fallback will be the default value of the
    stimulus parameters object for these parameters anytime the value passed in is `None`.
    """


    stim_param_val_combos = itertools.product(
        multi_params.spat_freqs,
        multi_params.temp_freqs,
        multi_params.orientations,
        multi_params.contrasts,
        multi_params.amplitudes,
        multi_params.DC_vals
        )
    stim_param_combos = tuple(
        do.GratingStimulusParams(
            spat_freq=SpatFrequency(c[0], multi_params.spat_freq_unit),
            temp_freq=TempFrequency(c[1], multi_params.temp_freq_unit),
            orientation=ArcLength(c[2], multi_params.ori_arc_unit),
            # Use default value as the fall back if value is None
            contrast=(do.ContrastValue(c[3])
                if c[3] is not None
                else do.GratingStimulusParams.__dataclass_fields__['contrast'].default
                ),
            amplitude=(c[4]
                if c[4] is not None
                else do.GratingStimulusParams.__dataclass_fields__['amplitude'].default
                ),
            DC=(c[5]
                if c[5] is not None
                else do.GratingStimulusParams.__dataclass_fields__['DC'].default
                ),
            )
        for c in  stim_param_val_combos
    )

    return stim_param_combos


def mk_stimulus_cache(
        st_params: do.SpaceTimeParams,
        multi_stim_params: do.MultiStimulusParams,
        overwrite: bool = False
        ):
    """Create stimulus arrays and save to file for all parameters provided

    Examples:
        >>> mk_stimulus_cache(stparams,
                spat_freqs=[2,4], temp_freqs=[1,2], orientations=[0, 90],
                spat_freq_unit='cpd', temp_freq_unit='hz', ori_arc_unit='deg',
                overwrite=True
            )
    """

    stim_param_combos = multi_stim_params

    cached_stim_params = get_params_for_all_saved_stimuli()
    # space time params already exist ... maybe some stim params will too?
    if st_params in cached_stim_params:
        for stim_params in stim_param_combos:
            # exists and no overwriting ... skip
            if (stim_params in cached_stim_params[st_params]) and (not overwrite):
                print(f'Skipping {stim_params} \n ... as exists and overwrite not set')
                continue
            # doesn't exist ... OR overwrite anyway ... make and save
            elif (stim_params not in cached_stim_params[st_params]) or (overwrite):
                mk_save_stim_array(st_params, stim_params)
    # need to just write it all out
    else:
        print(f'writing stimuli to file ...\n')
        for i, stim_params in enumerate(stim_param_combos):
            print(f'writing {i:<4} / {len(stim_param_combos):<4}', end='\r')
            mk_save_stim_array(st_params, stim_params)


def mk_rf_stim_spatial_slice_idxs(
        st_params: do.SpaceTimeParams,
        spat_filt_params: do.DOGSF,
        spat_filt_location: do.RFLocation
        ) -> do.RFStimSpatIndices:



    spat_res = st_params.spat_res
    # snap location to spat_res grid
    sf_loc = spat_filt_location.round_to_spat_res(spat_res)

    # spatial filter's size in number of spatial resolution steps (ie, index vals)
    sf_ext_n_res = ff.spatial_extent_in_res_units(spat_res, sf=spat_filt_params)
    # need radius, as will start with location as the center, then +/- the radius
    # use truncated half (//2) as guaranteed to be odd with a central value
    # that will be the location, radius will then "added" as a negative and positive
    # extension of the central location coordinate
    sf_radius_n_res = sf_ext_n_res//2

    # stimulus/canvas's spatial extent
    stim_ext_n_res = ff.spatial_extent_in_res_units(spat_res, spat_ext=st_params.spat_ext)
    # index that corresponds to the center of the spatial coordinates
    # as guaranteed to be odd, floor(size/2) or size//2 is the center index
    stim_cent_idx = int(stim_ext_n_res//2)  # as square, center is same for x and y

    # spatial filter location indices.
    # As guaranteed to be snapped to res,
    # should be whole number quotients when divided by spat_res raw value
    x_pos_n_res = sf_loc.x.value // spat_res.value
    y_pos_n_res = sf_loc.y.value // spat_res.value

    # slicing indices
    # 1: start with center (as all location coordinates treat the center as 0,0)
    # 2: subtract the spatial filter radius for starting index
    # 3: add spatial filter location index to translate the slice so that the sf's location
    #     will be the center of the slice
    # 4: ADD instead of subtract the radius, as this is the endpoint index of the slice
    # 5: translate for the location as before, where location coordinates treat the center
    #     as (0,0) and can be either pos or neg, so in either case, just need to add
    # 6: add 1 to endpoint index as in python this is not inclusive, so, to get the full radius
    #     on both sides we need to add 1 to the endpoint.
    #     EG, if `3` is the location idx and the radius is `2`:
    #       a[3-2:3] has size `2` but does not include the center/location (endpoint is not inclusive)
    #       a[3:3+2] also has size `2`, but includes the center, so we're missing the final value
    #       to make up the full radial extension of 3 values out from the center.
    #     What we want is [2 vals] + [center] + [2 vals].  a[3-2:3] gives us "[2 vals]".
    #      But, a[3:3+2] gives us [center] + [1 vals].  We need that last remaining value,
    #      thus ... "+1".
    slice_x_idxs = (
        #   1               2                 3
        int(stim_cent_idx - sf_radius_n_res + x_pos_n_res),
        #                   4                 5             6
        int(stim_cent_idx + sf_radius_n_res + x_pos_n_res + 1),
        )

    # y coord slices ... same as for x but subtract the location indices
    # as the first rows of the array are treated as the "top" of the "image".
    # Thus, positive y coordinates represent the first rows (with minimal/low indices)
    #  and negative y coords represent the last rows (maximal/high indices)
    slice_y_idxs = (
        int(stim_cent_idx - sf_radius_n_res - y_pos_n_res),
        int(stim_cent_idx + sf_radius_n_res - y_pos_n_res + 1),
        )

    rf_idxs = do.RFStimSpatIndices(
        x1=slice_x_idxs[0], x2=slice_x_idxs[1],
        y1=slice_y_idxs[0], y2=slice_y_idxs[1])

    return rf_idxs


def mk_stimulus_slice_array(
        st_params: do.SpaceTimeParams,
        stimulus_array: np.ndarray,
        slice_idxs: do.RFStimSpatIndices
        ) -> np.ndarray:

    # require that slice is within the bounds of the sitmulus array
    if not slice_idxs.is_within_extent(st_params):
        raise exc.LGNError('Stimulus slice indices are not within bounds of stimulus')

    sliced_array = stimulus_array[slice_idxs.y1:slice_idxs.y2, slice_idxs.x1:slice_idxs.x2]


    return sliced_array




