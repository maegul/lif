"""Create the cells for an LGN layer to make an lgn layer object

Examples:
    # +
    stparams = do.SpaceTimeParams(
        spat_ext=ArcLength(5), spat_res=ArcLength(1, 'mnt'), temp_ext=Time(1),
        temp_res=Time(1, 'ms'))

    lgnparams = do.LGNParams(
        n_cells=10,
        orientation = do.LGNOrientationParams(ArcLength(30), 0.5),
        circ_var = do.LGNCircVarParams('naito_lg_highsf'),
        spread = do.LGNLocationParams(2, 'jin_etal_on'),
        filters = do.LGNFilterParams(spat_filters='all', temp_filters='all')
        )
    # -
    # +
    lgn = mk_lgn_layer(lgnparams, spat_res=stparams.spat_res)
    # -
    # +
    len(lgn.cells)
    # -
    # +
    for i in range(len(lgn.cells)):
        print(
            lgn.cells[i].location
            )
    # -
    # +
    for i in range(len(lgn.cells)):
        print(
            lgn.cells[i].location.round_to_spat_res(stparams.spat_res)
            )
    # -
"""
# # Imports
import random
from textwrap import dedent
from typing import List, cast, Dict, Tuple, Sequence, Optional, Callable, Any
from collections import abc
from itertools import combinations, combinations_with_replacement
import json
from dataclasses import dataclass

import numpy as np
from pandas.core.indexes.base import default_index

import plotly.express as px
from scipy.spatial.distance import pdist

from ..utils import data_objects as do, settings, exceptions as exc
from ..utils.units.units import (
    scalar,
    ArcLength, Time
    )
from ..receptive_field.filters import cv_von_mises as cvvm
from . import (
    rf_locations as rflocs,
    orientation_preferences as rforis)

from ..receptive_field.filters import filters
from ..receptive_field.filters import (
    filter_functions as ff,
    contrast_correction as cont_corr
    )

# # spatial and temporal filter dictionaries
spatial_filters = filters.spatial_filters
temporal_filters = filters.temporal_filters


# # Shortcuts to RF Loc Distributions
# they come from file ... just list them as done for spatial+temproal but here
# they're much more constrained ... so it shold maked sense without an index file

rfloc_dist_index_type = Dict[str, str]

def get_rfloc_dist_index() -> rfloc_dist_index_type:

    data_dir = settings.get_data_dir()
    index_path = data_dir / 'rfloc_dist_index.json'
    try:
        with open(index_path, 'r') as f:
            rfloc_dist_index = json.load(f)
    except Exception as e:
        raise exc.FilterError(f'Count not loda filter index at {index_path}') from e

    return rfloc_dist_index


def get_rfloc_dists(
        index: rfloc_dist_index_type
        ) -> Dict[str, do.RFLocationSigmaRatio2SigmaVals]:

    rfdists = {}
    for dist_alias, file_name in index.items():
        try:
            new_rfdist = do.RFLocationSigmaRatio2SigmaVals.load(file_name)
        except Exception as e:
            raise exc.LGNError(f'Could not load rf dist at {file_name}') from e
        rfdists[dist_alias] = new_rfdist

    return rfdists

# ## Load RF Dists and assign to module variables
rf_dists = get_rfloc_dists(get_rfloc_dist_index())


# # functions for generating cells

# ## Make Orientation Values

def mk_orientations(
        n: int,
        ori_params: do.LGNOrientationParams,
        ) -> List[ArcLength[scalar]]:

    angles, probs = rforis._mk_angle_probs(ori_params.von_mises)
    random_angles = np.random.choice(angles.deg, p=probs, size=n)
    orientations = [
        ArcLength(a, 'deg')
        for a in random_angles
    ]

    return orientations

# ## Make Circular Variance Values

def mk_circ_var_values(
        n: int, cv_params: do.LGNCircVarParams
        ) -> np.ndarray:

    cv_dist = (
              rforis
              .circ_var_distributions
              .get_distribution(cv_params.distribution_alias)
              )
    cvvals = cv_dist.distribution.rvs(size=n)
    # Clip to [0,1]
    cvvals[cvvals<0] = 0
    cvvals[cvvals>1] = 1

    return cvvals




# ## Make RF Locations

# +
def mk_rf_locations(
        spat_res: ArcLength[scalar],
        spat_filts: Sequence[do.DOGSpatialFilter],
        rf_loc_params: do.LGNLocationParams,
        pairwise_distance_coefficient: Optional[float] = None,
        rf_dist_scale: Optional[ArcLength[scalar]] = None
        ) -> do.LGNRFLocations:
    """Generate x,y coordinates for RF Locations

    Either pairwise_distance_coefficient or rf_dist_scale must be provided.
    But not both.
    Whichever is provided determines which method of scaling is employed

    """
    # One but not both
    assert (
        (pairwise_distance_coefficient or rf_dist_scale)
        and not
        (pairwise_distance_coefficient and rf_dist_scale)
        ), 'Must provide only one of "pairwise_distance_coefficient" and "rf_dist_scale"'

    n = len(spat_filts)

    rf_loc_gen = rf_dists.get(rf_loc_params.distribution_alias)
    if not rf_loc_gen:
        raise exc.LGNError(f'bad rf loc dist alias ({rf_loc_params.distribution_alias})')

    x_locs, y_locs = rflocs.mk_unitless_rf_locations(
                        n=n, rf_loc_gen=rf_loc_gen,
                        ratio = rf_loc_params.ratio
                        )
    if rf_loc_params.rotate_locations:
        x_locs, y_locs = rflocs.rotate_rf_locations(
            x_locs, y_locs, orientation=rf_loc_params.orientation)

    if not (x_locs.shape == y_locs.shape):
        raise exc.LGNError(
            f'X and Y location coordinates are not of the same size ({x_locs.shape, y_locs.shape})')


    # scale by each spat_filts size

    if pairwise_distance_coefficient:
        coords_for_target_magnitude = rflocs.mk_spat_filt_coords_at_target_magnitude(
                spat_filts = spat_filts,
                magnitude_ratio = None,  # rely on settings
                spat_res=spat_res, round=True
            )
        # use a list below ... relying on python dict insertion order guarantee!

        # at this point, where the size of the spat filt is used to scale the location
        # the locations are no longer random but bound to the spatial filter that occurs in the same
        # location in the sequence passed in as an argument
        location_coords = tuple(
            (
                do.RFLocation(
                    x=ArcLength(x * (scale.value * pairwise_distance_coefficient), scale.unit),
                    y=ArcLength(y * (scale.value * pairwise_distance_coefficient), scale.unit)
                    )
            )
            for x,y, scale in zip(x_locs, y_locs, coords_for_target_magnitude)
        )

    # or, if provided, use the general rf_dist_scale

    elif rf_dist_scale:
        location_coords = tuple(
            (
                do.RFLocation(
                    x=ArcLength(x * rf_dist_scale.value, rf_dist_scale.unit),
                    y=ArcLength(y * rf_dist_scale.value, rf_dist_scale.unit)
                    )
            )
            for x,y in zip(x_locs, y_locs)
        )

    # given assert at the top, it should always be defined
    location_coords = cast(Tuple[do.RFLocation], location_coords)  # type: ignore

    rf_locations = do.LGNRFLocations(locations = location_coords)

    return rf_locations

# -


def calculate_max_spatial_ext_of_all_spatial_filters(
        spat_res: ArcLength[int],
        lgn_params: do.LGNParams,
        safety_margin_increment: float = 0
        ) -> ArcLength[scalar]:


    max_spat_ext = (
        2  # to make an extent and not radial
        *
        settings.simulation_params.spat_filt_sd_factor
        *
        max(
            (sf.parameters.max_sd().mnt
                for key, sf in spatial_filters.items()
                if (
                    # if not a list, than must be "all", so use this filter
                    not isinstance(lgn_params.filters.spat_filters, abc.Sequence)
                    or
                    # if is a sequence, then check if key is in provided list
                    key in lgn_params.filters.spat_filters
                    )
                ) )
    )

    max_spat_ext = ff.round_coord_to_res(
        ArcLength(max_spat_ext * (1 + safety_margin_increment), 'mnt'),
        spat_res, high=True
        )

    # use unit of 'mnt' as that's used in the calculation above for `max_sd()`.
    return max_spat_ext


# ## Make filters

def mk_filters(
        n: int, filter_params: do.LGNFilterParams
        ) -> Tuple[Tuple[do.DOGSpatialFilter, ...], Tuple[do.TQTempFilter, ...]]:
    """For each cell spat and temp filters randomly sampled from provided candidates
    """

    sfparams = filter_params.spat_filters
    tfparams = filter_params.temp_filters

    # just use all keys
    if sfparams == 'all':
        sfparams = tuple(spatial_filters.keys())
    # make sure all keys are valid, only need to check if not just taking all
    else:
        all_valid_keys = all(
            sf_key in spatial_filters
                for sf_key in sfparams
            )
        if not all_valid_keys:
            raise exc.LGNError(dedent(f'''
                Invalid spatial filter alias provided.
                Invalid: {(sfk for sfk in sfparams if sfk not in spatial_filters)}.
                Availabile options are {spatial_filters.keys()}
                '''))

    sf_keys = random.choices(sfparams, k=n)
    sfs = tuple(spatial_filters[sfk] for sfk in sf_keys)

    if tfparams == 'all':
        tfparams = tuple(temporal_filters.keys())
    else:
        all_valid_keys = all(
            tf_key in temporal_filters
                for tf_key in tfparams
            )
        if not all_valid_keys:
            raise exc.LGNError(dedent(f'''
                Invalid temporal filter alias provided.
                Invalid: {(tfk for tfk in tfparams if tfk not in temporal_filters)}.
                Availabile options are {temporal_filters.keys()}
                '''))

    tf_keys = random.choices(tfparams, k=n)
    tfs = tuple(temporal_filters[tfk] for tfk in tf_keys)

    return sfs, tfs


# ## Make Max F1 Amplitudes

def mk_max_f1_amplitudes(
        n: int,
        f1_amp_params: do.LGNF1AmpDistParams,
        contrast: Optional[do.ContrastValue] = None,
        contrast_params: Optional[do.ContrastParams] = None
        ) -> Sequence[do.LGNF1AmpMaxValue]:
    """Draw max F1 amplitudes from distribution defined by `f1_amp_params`.

    If `contrast` provided, the values will be contrast corrected
    """

    f1_amps = f1_amp_params.draw_f1_amp_vals(n = n)

    if contrast:
        # get default contrast params if missing
        if not contrast_params:
            contrast_params = settings.simulation_params.contrast_params

        contrast_adjusted_f1_amps = tuple(
            do.LGNF1AmpMaxValue(
                max_amp = cont_corr.correct_contrast_response_amplitude(
                    response_amplitude=max_f1_amp.max_amp,
                    base_contrast=max_f1_amp.contrast,
                    target_contrast=contrast,
                    contrast_params=contrast_params
                ),
                contrast = contrast
            )
            for max_f1_amp in f1_amps
            )

        return contrast_adjusted_f1_amps


    return f1_amps


# ## Make Cells (final stage)

def mk_lgn_layer(
        lgn_params: do.LGNParams,
        spat_res: ArcLength[scalar],
        contrast: Optional[do.ContrastValue] = None,
        use_dist_scale: bool = True,
        use_spat_filt_size_coefficient: bool = False,
        force_central: bool = False,
        rf_dist_scale_func: Optional[Callable] = None
        ) -> do.LGNLayer:
    """Make full lgn layer from params

    `contrast` necessary to correct the amplitude of the target F1 amplitudes.
    As this will shift the distribution of actual firing rates of the LGN cells, this is quite
    an important parameter in the simulation, as **this is where the actual contrast** of the
    simulation is defined and where it affects the LGN layer's response rates.

    `force_central`, if `True` will artificially place all RF Locations at `0,0`.
    """
    n_cells = lgn_params.n_cells

    # orientations
    orientations = mk_orientations(
        n = n_cells, ori_params = lgn_params.orientation)

    # circular variance (of spat filt orientation bias)
    circ_var_vals: List[float] = list(mk_circ_var_values(n_cells, lgn_params.circ_var))

    # spat_filters
    spat_filts, temp_filts = mk_filters(n_cells, lgn_params.filters)

    f1_max_amps = mk_max_f1_amplitudes(
        n_cells, lgn_params.F1_amps,
        contrast=contrast  # contrast provided for contrast correction
        )

    # oriented spat filters
    # let's pre-compute them and have both available
    # ... only a few params and objects more
    oriented_spat_filts: Tuple[do.DOGSpatFiltArgs,...] = tuple(
        ff.mk_ori_biased_spatfilt_params_from_spat_filt(
                spat_filt, circ_var,
                method=lgn_params.circ_var.circ_var_definition_method
                )
            for spat_filt, circ_var in zip(spat_filts, circ_var_vals)
        )

    # locations
    # artificially put all RF locations at 0 by using the distance_scale, as it is multiplied
    if force_central:
        locations = tuple(
                do.RFLocation(x=ArcLength(0), y=ArcLength(0))
                    for _ in range(n_cells)
            )
        rf_distance_scale = None  # as overwritten
        scale_val = rf_distance_scale
        rf_locations = do.LGNRFLocations(locations)

    else:
        if rf_dist_scale_func:
            rf_distance_scale = rf_dist_scale_func(spat_filts, spat_res)
            scale_val = rf_distance_scale

            rf_locations = mk_rf_locations(spat_res=spat_res,
                spat_filts = spat_filts,
                rf_loc_params=lgn_params.spread,
                rf_dist_scale=rf_distance_scale
                )

        elif use_dist_scale:
            rf_distance_scale = rflocs.mk_rf_locations_distance_scale(
                spat_filters=spat_filts, spat_res=spat_res,
                # This uses the mean ... more correlated to actual SFs and their sizes
                # should smooth out irregular scaling of distance
                use_median_for_pairwise_avg=False,
                magnitude_ratio_for_diameter=None  # relying on default value in settings
                )
            scale_val = rf_distance_scale

            rf_locations = mk_rf_locations(spat_res=spat_res,
                spat_filts = spat_filts,
                rf_loc_params=lgn_params.spread,
                rf_dist_scale=rf_distance_scale
                )

        elif use_spat_filt_size_coefficient:
            scale_val = 2.55  # just hardcoding for now
            rf_locations = mk_rf_locations(spat_res=spat_res,
                spat_filts = spat_filts,
                rf_loc_params=lgn_params.spread,
                pairwise_distance_coefficient=scale_val
                )
        else:
            raise ValueError('must use either rf_dist_scale or pairwise size coefficient')

    lgn_cells = tuple(
            do.LGNCell(
                    spat_filts[i],
                    oriented_spat_filts[i],
                    temp_filts[i],
                    f1_max_amps[i],
                    orientations[i],
                    circ_var_vals[i],
                    rf_locations.locations[i],
                    )
            for i in range(lgn_params.n_cells)
        )

    lgn_layer = do.LGNLayer(cells=lgn_cells, params=lgn_params,
        rf_distance_scale=scale_val
        )

    return lgn_layer



# # Recreating Objects from records of cell/lgn objects

# could make a protocol to type annotate for a dataclass instance ... but not necessary here
def _mk_dataclass_fields_dict(dataclass_instance) -> Dict[str, Any]:
    """dict of only the first level of fields/parameters"""

    try:
        fields = dataclass_instance.__dataclass_fields__.keys()
    except AttributeError as e:
        raise ValueError('passed argument is not an instance of a dataclass') from e

    parameters = {f: dataclass_instance.__getattribute__(f) for f in fields}

    return parameters


def mk_lgn_cell_record(lgn_cell: do.LGNCell) -> do.LGNCellRecord:

    # main thing is to avoid saving the whole filter object as it carries a bit
    # ... of baggage ... so just store the keys and load from file/(filters module)
    # ... when necessary.

    parameters = _mk_dataclass_fields_dict(lgn_cell)
    parameters['spat_filt'] = lgn_cell.spat_filt.key
    parameters['temp_filt'] = lgn_cell.temp_filt.key

    record = do.LGNCellRecord(**parameters)

    return record

def mk_cell_from_record(record: do.LGNCellRecord) -> do.LGNCell:

    parameters = _mk_dataclass_fields_dict(record)
    parameters['spat_filt'] = spatial_filters[filters.reverse_spatial_filters[record.spat_filt]]
    parameters['temp_filt'] = temporal_filters[filters.reverse_temporal_filters[record.temp_filt]]

    cell = do.LGNCell(**parameters)

    return cell


def mk_lgn_layer_record(lgn_layer: do.LGNLayer) -> do.LGNLayerRecord:
    """
    using an LGN record that reduces the size on disk by storing only a reference/key
    to each RF rather than the whole thing as there are only a finite set and they are
    already stored separately
    """

    cell_records = tuple(
        mk_lgn_cell_record(c)
        for c in lgn_layer.cells
        )
    parameters = _mk_dataclass_fields_dict(lgn_layer)
    parameters['cells'] = cell_records

    record = do.LGNLayerRecord(**parameters)

    return record


def mk_lgn_layer_from_record(record: do.LGNLayerRecord) -> do.LGNLayer:

    parameters = _mk_dataclass_fields_dict(record)
    cells = tuple(
            mk_cell_from_record(cell_record)
            for cell_record in parameters['cells']
        )
    parameters['cells'] = cells

    lgn_layer = do.LGNLayer(**parameters)

    return lgn_layer


def mk_contrast_lgn_layer_collection_record(
        lgn_collection: do.ContrastLgnLayerCollection
        ) -> do.ContrastLgnLayerCollectionRecord:

    record = {
        contrast: tuple(
            mk_lgn_layer_record(layer) for layer in layers
            )
        for contrast, layers in lgn_collection.items()
    }

    return record


def mk_contrast_lgn_layer_collection_from_record(
        lgn_collection_record: do.ContrastLgnLayerCollectionRecord
        ) -> do.ContrastLgnLayerCollection:

    layer_collection = {
        contrast: tuple(
            mk_lgn_layer_from_record(layer_record) for layer_record in layer_records
            )
        for contrast, layer_records in lgn_collection_record.items()
    }

    return layer_collection
