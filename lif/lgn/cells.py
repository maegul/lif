# > Imports
from typing import List, cast, Dict
import json
from dataclasses import dataclass

import numpy as np

import plotly.express as px

from ..utils import data_objects as do, settings, exceptions as exc
from ..utils.units.units import (
    scalar,
    ArcLength
    )
from ..receptive_field.filters import cv_von_mises as cvvm
from . import rf_locations, orientation_preferences

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



# > functions for generating cells

def mk_cells(params: do.LGNParams) -> do.LGNLayer:

    orientations = orientation_preferences.mk_orientations(
        n = params.n_cells, ori_params = params.orientation)

    return do.LGNLayer(cells=[])


# functions for generating random values
# what parameters to set (which distributions to use)

# list all sf and tf files (by string) and hard code variables for quick lookup
# which to use are parameters
