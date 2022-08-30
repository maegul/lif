from typing import Optional, Union, cast, Tuple, overload

import numpy as np

# import matplotlib.pyplot as plt

# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui

# import scipy.stats as st

# I'd prefer to move the coords functions to utls.coords.py
from ..receptive_field.filters import filter_functions as ff
from ..utils.units.units import SpatFrequency, TempFrequency, ArcLength, Time, val_gen
from ..utils import data_objects as do

# from . import data_objects as do, estimate_real_amp_from_f1 as est_amp

PI: float = np.pi


def mk_sinstim(
        space_time_params: do.SpaceTimeParams,
        grating_stim_params: do.GratingStimulusParams
        ) -> np.ndarray:
    '''Generate sinusoidal grating stimulus

    Returns 3D array: x, y, t

    Grating "lines" are oriented according to "orientation" in grating_stim_params
    Modulation or propogation over time is in "direction" in grating_stim_params

    Spatial and Temporal coords are generated according to space_time_params
    '''

    # creating coords
    xc, yc, tc = ff.mk_spat_temp_coords(
        spat_ext=space_time_params.spat_ext, temp_ext=space_time_params.temp_ext,
        spat_res=space_time_params.spat_res, temp_res=space_time_params.temp_res
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


def mk_rf_stim_spatial_slice_idxs(
        st_params: do.SpaceTimeParams,
        spat_filt_params: do.DOGSpatFiltArgs,
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
