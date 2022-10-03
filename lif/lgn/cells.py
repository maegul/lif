# > Imports
import statistics
from typing import List, cast, Dict, Tuple, Iterable
from itertools import combinations, combinations_with_replacement
import json
from dataclasses import dataclass

import numpy as np
from pandas.core.indexes.base import default_index

import plotly.express as px

from ..utils import data_objects as do, settings, exceptions as exc
from ..utils.units.units import (
    scalar,
    ArcLength
    )
from ..receptive_field.filters import cv_von_mises as cvvm
from . import rf_locations, orientation_preferences as oris

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

    angles, probs = oris._mk_angle_probs(ori_params.von_mises)
    random_angles = np.random.choice(angles.deg, p=probs, size=n)
    orientations = [
        ArcLength(a, 'deg')
        for a in random_angles
    ]

    return orientations

# >> Make Circular Variance Values

def mk_circ_var_values(
        n: int, cv_dist: do.CircVarianceDistribution
        ) -> List[float]:

    cvvals = [cv for cv in cv_dist.distribution.rvs(size=n)]
    return cvvals



# >> Make RF Locations
# +
def mk_unitless_rf_locations(
        n: int,
        rf_loc_params: do.LGNLocationParams,
        ):

    rf_loc_gen = rf_dists.get(rf_loc_params.distribution_alias)
    if not rf_loc_gen:
        raise exc.LGNError(f'bad rf loc dist alias ({rf_loc_params.distribution_alias})')

    gauss_params = rf_loc_gen.ratio2gauss_params(rf_loc_params.ratio)
    x_locs, y_locs = (
        np.random.normal(scale=s, size=n)
        for s in
            (gauss_params.sigma_x, gauss_params.sigma_y)
        )
    rotated_coords = rf_locations.rotate_rf_locations(
            np.vstack((x_locs, y_locs)).T,  # stacking into two columns (x,y)
            rf_loc_params.orientation)

    return (rotated_coords[:,0], rotated_coords[:,1])
# -
# +
def mk_rf_locations(
        n: int,
        rf_loc_params: do.LGNLocationParams,
        distance_scale: ArcLength[scalar],
        ) -> Tuple[Tuple[ArcLength, ArcLength]]:

    x_locs, y_locs = mk_unitless_rf_locations(n, rf_loc_params)

    rf_locations = tuple(
        (
            ArcLength(x_locs[i] * distance_scale.value, distance_scale.unit),
            ArcLength(y_locs[i] * distance_scale.value, distance_scale.unit)
        )
        for i in range(len(x_locs))
    )

    return rf_locations

# -


# >> Avg Biggest size of pair??
# seems to approach a ratio of ~2/3 as the set of values gets bigger
# ... probably best to just calculate from all available diameters
# +
max_diam = range(10, 100, 10)
avg_max_val_ratios = []
for n in max_diam:
    # n = 10
    diams = np.arange(1, n, 1)
    diam_combs = list(combinations_with_replacement(diams,r=2))
    pair_max_diams = [max(p) for p in diam_combs]

    max_val = max(diams)
    avg_pair_max, median_pair_max = np.mean(pair_max_diams), np.median(pair_max_diams)
    n_pairs = len(diam_combs)
    avg_max_ratio = avg_pair_max/max_val
    avg_max_val_ratios.append(avg_max_ratio)

    print(f'''
        {n=}, {n_pairs=}
        {avg_pair_max=}, {median_pair_max=}
        {avg_max_ratio=}
        ''')

# -
# +
px.line(x=max_diam, y=avg_max_val_ratios).show()
# -

# >> Make Cells

def mk_cells(params: do.LGNParams) -> do.LGNLayer:

    n_cells = params.n_cells

    orientations = mk_orientations(
        n = n_cells, ori_params = params.orientation)

    # >>! need to do spatial filters before RF locs for the spatial scale!
    # get all spatial filters
    # get max_sd values for each
    # find avg of biggest half (to correspond to jin et al methodology of biggest of pair)

    return do.LGNLayer(cells=[])


# functions for generating random values
# what parameters to set (which distributions to use)

# list all sf and tf files (by string) and hard code variables for quick lookup
# which to use are parameters
