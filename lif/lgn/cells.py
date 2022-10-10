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

# > Imports
import random
from textwrap import dedent
from typing import List, cast, Dict, Tuple, Iterable
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

from ..receptive_field.filters import filter_functions as ff


# > Shortcuts to spatial and temporal filters

filter_index = Dict[str, Dict[str, str]]

def get_filter_index() -> filter_index:

    data_dir = settings.get_data_dir()
    filter_index_path = data_dir / 'filter_index.json'
    try:
        with open(filter_index_path, 'r') as f:
            filter_index = json.load(f)
    except Exception as e:
        raise exc.FilterError(f'Count not loda filter index at {filter_index_path}') from e

    return filter_index


_filter_type_lookup = {
    'spatial': do.DOGSpatialFilter,
    'temporal': do.TQTempFilter
}


def get_filters(index: filter_index):

    filters = {}
    for filter_type, filter_items in index.items():
        filters[filter_type] = {}
        for filter_alias, file_name in filter_items.items():
            try:
                new_filter = _filter_type_lookup[filter_type].load(file_name)
            except Exception as e:
                raise exc.FilterError(f'Could not load filter at {file_name}') from e
            filters[filter_type][filter_alias] = new_filter

    return filters

# >> Load filters and set to module variables
_all_filters = get_filters(get_filter_index())
spatial_filters: Dict[str, do.DOGSpatialFilter] = _all_filters['spatial']
temporal_filters: Dict[str, do.TQTempFilter] = _all_filters['temporal']


# > Shortcuts to RF Loc Distributions
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

# >> Load RF Dists and assign to module variables
rf_dists = get_rfloc_dists(get_rfloc_dist_index())


# > functions for generating cells

# >> Make Orientation Values

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

# >> Make Circular Variance Values

def mk_circ_var_values(
        n: int, cv_params: do.LGNCircVarParams
        ) -> List[float]:

    cv_dist = (
              rforis
              .circ_var_distributions
              .get_distribution(cv_params.distribution_alias)
              )

    cvvals = [cv for cv in cv_dist.distribution.rvs(size=n)]
    return cvvals



# >> Make RF Locations

# +
def mk_rf_locations(
        n: int,
        rf_loc_params: do.LGNLocationParams,
        distance_scale: ArcLength[scalar],
        ) -> do.LGNRFLocations:
    """Generate x,y coordinates for RF Locations

    """

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

    location_coords = tuple(
        (
            do.RFLocation(
                x=ArcLength(x_locs[i] * distance_scale.value, distance_scale.unit),
                y=ArcLength(y_locs[i] * distance_scale.value, distance_scale.unit)
                )
        )
        for i in range(len(x_locs))
    )
    rf_locations = do.LGNRFLocations(locations = location_coords)

    return rf_locations

# -

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



# >> Make Cells

def mk_lgn_layer(
        lgn_params: do.LGNParams,
        spat_res: ArcLength[scalar],
        ) -> do.LGNLayer:

    n_cells = lgn_params.n_cells

    # orientations
    orientations = mk_orientations(
        n = n_cells, ori_params = lgn_params.orientation)

    # circular variance (of spat filt orientation bias)
    circ_var_vals = mk_circ_var_values(n_cells, lgn_params.circ_var)

    # spat_filters
    spat_filts, temp_filts = mk_filters(n_cells, lgn_params.filters)

    # oriented spat filters
    # let's pre-compute them and have both available
    # ... only a few params and objects more
    oriented_spat_filts = tuple(
        ff.mk_ori_biased_spatfilt_params_from_spat_filt(
                spat_filt, circ_var,
                method=lgn_params.circ_var.circ_var_definition_method
                )
            for spat_filt, circ_var in zip(spat_filts, circ_var_vals)
        )

    # locations
    rf_distance_scale = rflocs.mk_rf_locations_distance_scale(
        spat_filters=spat_filts, spat_res=spat_res,
        # >>>! should be in settings?
        magnitude_ratio_for_diameter=0.2
        )
    rf_locations = mk_rf_locations(
        n=n_cells, rf_loc_params=lgn_params.spread,
        distance_scale=rf_distance_scale
        )


    lgn_layer = do.LGNLayer(
        cells = tuple(
            do.LGNCell(
                orientation=ori, circ_var=cv,
                spat_filt=sf, oriented_spat_filt_params=ori_sf,
                temp_filt=tf,
                location=loc
                )
            for ori, cv, sf, ori_sf, tf, loc
                in zip(
                    orientations,
                    circ_var_vals,
                    spat_filts, oriented_spat_filts,
                    temp_filts,
                    rf_locations.locations)
            ),
        params = lgn_params
    )

    return lgn_layer
