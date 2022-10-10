from typing import cast
import numpy as np

from pytest import mark, raises
import hypothesis
from hypothesis import given, strategies as st, assume, event

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

from lif.lgn import (
    rf_locations as rflocs,
    orientation_preferences as rforis,
    cells
    )

from lif.stimulus import stimulus as stim


# > RF Locations
try:
    rf_loc_gen = cells.rf_dists.get('jin_etal_on')
except Exception as e:
    raise ValueError('cannot run test as data files for rf dists not correct') from e

rf_loc_gen = cast(do.RFLocationSigmaRatio2SigmaVals, rf_loc_gen)

# >> Rotation and Pairwise distance constancy
@mark.integration
@given(
    rotation_angle_value=st.floats(0, 90, allow_infinity=False, allow_nan=False),
    distance_scale_value=st.floats(0, 1000, allow_infinity=False, allow_nan=False),
    location_sigma_ratio=st.floats(1.1, 20, allow_infinity=False, allow_nan=False),
    )
def test_pairwise_dists_constant_under_rotation(
        location_sigma_ratio: float,
        distance_scale_value: float, rotation_angle_value: float,
        ):

    # get an rf loc generator (relies on data files being saved and correct unfortunately)

    distance_scale = ArcLength(distance_scale_value, 'deg')
    rotation_angle = ArcLength(rotation_angle_value, 'deg')


    # >>> rotation doesn't change pw dists

    x_locs, y_locs = rflocs.mk_unitless_rf_locations(
        10, rf_loc_gen, ratio=location_sigma_ratio)

    rot_x_locs, rot_y_locs = rflocs.rotate_rf_locations(
        x_locs, y_locs, rotation_angle)

    assert np.allclose(
        np.sort(rflocs.pdist(rflocs.mk_coords_2d(x_locs, y_locs))),
        np.sort(rflocs.pdist(rflocs.mk_coords_2d(rot_x_locs, rot_y_locs)))
        )

    # >>> Scaling and Rotation are commutative
    # apply distance scale then rotate
    scaled_locs = rflocs.apply_distance_scale_to_rf_locations(
        x_locs, y_locs, distance_scale)

    scaled_rotated_x_locs, scaled_rotated_y_locs = rflocs.rotate_rf_locations(
        *scaled_locs.array_of_coords(), ArcLength(45, 'deg')
        )

    # rotate then apply distance scale
    rot_x_locs, rot_y_locs = rflocs.rotate_rf_locations(
        x_locs, y_locs, rotation_angle)

    rotated_scaled_x_locs, rotated_scaled_y_locs = (
        rflocs
        .apply_distance_scale_to_rf_locations(
            rot_x_locs, rot_y_locs, distance_scale
        )
        .array_of_coords()
    )


    assert np.allclose(
        np.sort(rflocs.pdist(rflocs.mk_coords_2d(
                                scaled_rotated_x_locs, scaled_rotated_y_locs))),
        np.sort(rflocs.pdist(rflocs.mk_coords_2d(
                                rotated_scaled_x_locs, rotated_scaled_y_locs)))
        )


# > Stimulus
fltst = st.floats(min_value=0, max_value=10_000, allow_nan=False, allow_infinity=False)
intst = st.integers(min_value=0, max_value=10_000)
arc_units = st.sampled_from(('deg', 'mnt', 'sec', 'rad'))
tim_units = st.sampled_from(('s', 'ms', 'us'))
sfreq_units = st.sampled_from(('cpd', 'cpm', 'cpd_w'))
tfreq_units = st.sampled_from(('hz', 'w'))

@mark.proto
@given(
    spat_ext=fltst, spat_res=intst, temp_ext=fltst, temp_res=intst,
    spat_ext_unit=arc_units, spat_res_unit=arc_units, temp_ext_unit=tim_units,
    temp_res_unit=tim_units,
    spat_freq=fltst, temp_freq=fltst, orientation=fltst, amplitude=fltst, DC=fltst,
    spat_freq_unit=sfreq_units, temp_freq_unit=tfreq_units, orientation_unit=arc_units
    )
def test_stimulus_signature(
    spat_ext, spat_res, temp_ext, temp_res,
    spat_ext_unit, spat_res_unit, temp_ext_unit, temp_res_unit,
    spat_freq, temp_freq, orientation, amplitude, DC,
    spat_freq_unit, temp_freq_unit, orientation_unit
    ):

    stparams = do.SpaceTimeParams(
        spat_ext=ArcLength(spat_ext, spat_ext_unit),
        spat_res=ArcLength(spat_res, spat_res_unit),
        temp_ext=Time(temp_ext, temp_ext_unit),
        temp_res=Time(temp_res, temp_res_unit)
        )
    stimparams = do.GratingStimulusParams(
        spat_freq=SpatFrequency(spat_freq, spat_freq_unit),
        temp_freq=TempFrequency(temp_freq, temp_freq_unit),
        orientation=ArcLength(orientation, orientation_unit),
        amplitude=amplitude, DC=DC
        )

    signature = stim.mk_stim_signature(stparams, stimparams)
    new_stparams, new_stimparams = stim.mk_params_from_stim_signature(signature)
    new_signature = stim.mk_stim_signature(new_stparams, new_stimparams)

    assert new_stparams == stparams
    assert new_stimparams == stimparams
    assert signature == new_signature

