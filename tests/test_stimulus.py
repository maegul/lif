# > imports
from pathlib import Path
import re
from typing import cast, Tuple
from dataclasses import astuple, asdict

from pytest import mark, raises
import hypothesis
from hypothesis import given, strategies as st, assume, event

from lif.stimulus import stimulus as stim

from lif.convolution import (
    convolve as conv,
    estimate_real_amp_from_f1 as est_amp
    )
from lif.utils.units.units import ArcLength, SpatFrequency, TempFrequency, Time, scalar
from lif.utils import (
    settings,
    data_objects as do,
    exceptions as exc
    )

from lif.receptive_field.filters import (
    filters,
    filter_functions as ff,
    cv_von_mises as cvvm
    )

import numpy as np
from dataclasses import dataclass
from numpy import fft  # type: ignore
from scipy.signal import convolve, argrelmin
# Test orientation and direction correct

# > Using Stimulus for spatial filters

@dataclass
class DummySFParams:
    spat_sd: ArcLength[scalar]

    def max_sd(self):
        return self.spat_sd

    @property
    def parameters(self):
        return self


# >> Slicing
@mark.proto
@mark.integration
@given(
        # a little fine tuning here to make sure the RFs fit into the
        # coordinates without generating massive arrays
        # presuming an sd factor of 5, the rf_ext_divisor, say 30, creates an ext of
        # (spat_ext/30)*5 * 2 (2 as std, which is both pos and negative)
        # which is 1/3 spat_ext (at greatest).
        # the rf_loc values, at most, are 20/100 = 1/5 of ext ...
        # so should always be within ext.
        # Note, there is a check for this and a log to the events ...
        spat_res=st.integers(1, 10),
        spat_res_unit=st.sampled_from(['mnt', 'sec']),  # deg would be too big to be integer
        spat_ext=st.floats(100, 200, allow_infinity=False, allow_nan=False),
        spat_ext_unit=st.sampled_from(['mnt', 'deg']),  # sec would be too small
        rf_ext_divisor=st.floats(30, 50, allow_infinity=False, allow_nan=False),
        rf_loc_x=st.floats(0, 20, allow_infinity=False, allow_nan=False),
        rf_loc_y=st.floats(0, 20, allow_infinity=False, allow_nan=False),
    )
def test_spatial_stim_slicing(
        spat_res, spat_res_unit,
        spat_ext, spat_ext_unit,
        rf_ext_divisor,
        rf_loc_x, rf_loc_y
        ):

    spat_res = ArcLength(
            int(ArcLength(spat_res, 'mnt')[spat_res_unit]),
            spat_res_unit
        )

    # ensure values are kept small enough to not blow up
    # the size of arrays
    # applies to all units that use spat_ext_unit
    if spat_ext_unit == 'deg':
        spat_ext, rf_loc_x, rf_loc_y = (
            # not 60, but 55, to create more random vals
            v / 55 for v in (
            spat_ext, rf_loc_x, rf_loc_y
            ) )

    spat_ext = ArcLength(spat_ext, spat_ext_unit)
    st_params = do.SpaceTimeParams(
        spat_res=spat_res, spat_ext=spat_ext,
        temp_res=Time(1), temp_ext=Time(1))

    rf_ext = ArcLength(spat_ext.value / rf_ext_divisor, spat_ext_unit)
    dummy_rf_args = DummySFParams(spat_sd=rf_ext)
    dummy_rf_args = cast(do.DOGSpatFiltArgs, dummy_rf_args)

    rf_loc = do.RFLocation(
        x=ArcLength(rf_loc_x, spat_ext_unit),
        y=ArcLength(rf_loc_y, spat_ext_unit)
        ).round_to_spat_res(st_params.spat_res)

    # event(f'{spat_res.value, spat_res.unit} ... {spat_ext.value, spat_ext.unit}')
    xc, yc = ff.mk_spat_coords(spat_res=st_params.spat_res, spat_ext=st_params.spat_ext)
    rf_idxs = stim.mk_rf_stim_spatial_slice_idxs(
        st_params=st_params,
        spat_filt_params=dummy_rf_args,
        spat_filt_location=rf_loc
        )

    if rf_idxs.is_within_extent(st_params):

        event(f'Indices within extent')
        # event(f'Indices within extent: {rf_ext_divisor}: {spat_ext.value}, {rf_ext.value} ({spat_ext.unit})')

        # y slices go first as first axis is rows which are is the y dimension
        #                               V: y first             V: x second
        stim_slice_x = xc.value[rf_idxs.y1:rf_idxs.y2, rf_idxs.x1:rf_idxs.x2]
        stim_slice_y = yc.value[rf_idxs.y1:rf_idxs.y2, rf_idxs.x1:rf_idxs.x2]

        stim_slice_x.shape
        stim_slice_x[0, 40:60]

        # >>> test position is accurate
        # x coords
        assert np.isclose(
            #            * any row or y coord would do, as they all have the same x coord vals
            #            |
            #            V                      V: floor(size/2) or size//2 is center coordinate
            stim_slice_x[0, stim_slice_x.shape[0]//2],
            rf_loc.x.value,  # should match THE ACTUAL SPATIAL VALUE as snapped to res vals
            )

        # y coords
        assert np.isclose(
            stim_slice_y[stim_slice_y.shape[0]//2 ,0],
            rf_loc.y.value
            )

        # >>> test size is accurate
        predicted_rf_size = ff.spatial_extent_in_res_units(
            st_params.spat_res, sf=dummy_rf_args)
        assert stim_slice_x.shape[0] == predicted_rf_size
        assert stim_slice_y.shape[0] == predicted_rf_size

    else:
        event(f'Not within extent: {spat_ext.value, rf_ext_divisor}({spat_ext.unit}) ... {rf_loc_x, rf_loc_y}')
    # assert False




    # predicted_n_res_units = ff.spatial_extent_in_res_units(spat_res, spat_ext=spat_ext)

    # assert xc.value.shape[0] == (predicted_n_res_units)
