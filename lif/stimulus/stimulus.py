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
    sf_loc = spat_filt_location.round_to_spat_res(spat_res)

    sf_ext_n_res = ff.spatial_extent_in_res_units(spat_res, sf=spat_filt_params)
    sf_radius_n_res = sf_ext_n_res//2
    stim_ext_n_res = ff.spatial_extent_in_res_units(spat_res, spat_ext=st_params.spat_ext)
    stim_cent_idx = int(stim_ext_n_res//2)  # as square, center is same for x and y
    # as snapped to res, should be whole number quotients
    x_pos_n_res = sf_loc.x.value // spat_res.value
    y_pos_n_res = sf_loc.y.value // spat_res.value

    slice_x_idxs = (
        int(stim_cent_idx - sf_radius_n_res + x_pos_n_res),
        # add 1 as endpoint of a slice is not inclusive (a[3:4] has size one!)
        int(stim_cent_idx + sf_radius_n_res + x_pos_n_res + 1),
        )
    slice_y_idxs = (
        int(stim_cent_idx - sf_radius_n_res - y_pos_n_res),
        int(stim_cent_idx + sf_radius_n_res - y_pos_n_res + 1),
        )

    rf_idxs = do.RFStimSpatIndices(
        x1=slice_x_idxs[0], x2=slice_x_idxs[1],
        y1=slice_y_idxs[0], y2=slice_y_idxs[1])

    return rf_idxs
