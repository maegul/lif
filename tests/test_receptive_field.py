from pathlib import Path
import re
from typing import cast, Tuple
from dataclasses import astuple, asdict

from pytest import mark, raises
import hypothesis
from hypothesis import given, strategies as st, assume, event

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
from numpy import fft  # type: ignore
from scipy.signal import convolve, argrelmin



def test_metadata_make_key_dt():
    meta_data = filters.do.CitationMetaData(**{
        'author': None, 'year': 1988, 'title': None, 'doi': 'address', 'reference': None})

    meta_data._set_dt_uid()

    assert meta_data.make_key() == meta_data._dt_uid.strftime('%y%m%d-%H%M%S')


@mark.parametrize(
    'metadata,expected',
    [
        ({'author': 'errol', 'year': 1988, 'title': 'test', 'doi': 'address', 'reference': None},
            'errol_1988_test'),
        ({'author': 'errol', 'year': 1988, 'title': 'test', 'doi': 'address', 'reference': 'fig3'},
            'errol_1988_test_fig3'),
        ({'author': 'errol', 'year': None, 'title': None, 'doi': 'address', 'reference': None},
            'errol'),
        ({'author': None, 'year': None, 'title': 'test', 'doi': 'address', 'reference': None},
            'test'),
        ({'author': None, 'year': None, 'title': 'test', 'doi': 'address', 'reference': 'fig3'},
            'test_fig3')
    ]
    )
def test_metadata_make_key(metadata: dict, expected: str):
    meta_data = filters.do.CitationMetaData(**metadata)

    key = meta_data.make_key()

    assert key == expected


# > Temporal

# >> Temp coords

def test_temp_coord_raises_exceptions():

    res = Time(1, 'ms')
    with raises(exc.CoordsValueError):
        ff.mk_temp_coords(res, Time(10), Time(10))
    with raises(exc.CoordsValueError):
        ff.mk_temp_coords(res, None, None)

def test_temp_coord_has_same_unit_as_res():
    units = ['s', 'us', 'ms']
    # res unit and extent unit are different, paired with the reverse of the other
    for ru, eu in zip(units, reversed(units)):
        coords = ff.mk_temp_coords(
            temp_res=Time(1., ru),
            temp_ext=Time(10., eu)
            )
        assert coords.unit == ru

@given(
    res=st.floats(0.1, 1, allow_nan=False, allow_infinity=False),
    ext_factor=st.floats(2, 1000, allow_nan=False, allow_infinity=False),
    res_unit=st.sampled_from(['us', 'ms', 's']),
    ext_unit=st.sampled_from(['us', 'ms', 's']))
def test_temp_coord_has_range_zero_to_ext(
        res, ext_factor, res_unit, ext_unit):

    temp_res = Time(res, res_unit)
    # ext made first in res units, then converted to ext_unit
    temp_ext_value = res * ext_factor
    temp_ext = Time(temp_ext_value, res_unit).in_same_units_as(Time(1,ext_unit))

    coords = ff.mk_temp_coords(temp_res, temp_ext)

    # min is zero
    assert np.isclose(coords.base[0], 0)

    # temp_ext <= max <= temp_ext +/ temp_res
    # use combo of isclose and a chained greater/equal to capture being either in range
    # or at the edges of the range (temp_ext - temp_ext+temp_res)
    assert (
        np.isclose(coords.base[-1], temp_ext.base, atol=1e-6)
        or
        np.isclose(coords.base[-1], temp_ext.base + temp_res.base, atol=1e-6)
        or
        (temp_ext.base <= coords.base[-1] <= (temp_ext.base + temp_res.base))
        )

@given(
    res=st.floats(0.1, 1, allow_nan=False, allow_infinity=False),
    tau=st.floats(10, 100, allow_nan=False, allow_infinity=False),
    res_unit=st.sampled_from(['us', 'ms', 's']),
    tau_unit=st.sampled_from(['us', 'ms', 's']),
    n_tau=st.integers(10, 30)
    )
def test_temp_coord_tau_ext(
        res: float, tau: float, res_unit: str, tau_unit: str, n_tau: int):

    temp_res = Time(res, res_unit)
    # temp_tau = Time(Time(tau, res_unit)[tau_unit], tau_unit)
    temp_tau = Time(tau, res_unit).in_same_units_as(Time(1, tau_unit))
    coords = ff.mk_temp_coords(
                temp_res, temp_ext=None,
                tau=temp_tau, temp_ext_n_tau=n_tau)

    total_tau = temp_tau.base * n_tau

    assert (
        np.isclose(coords.base[-1], total_tau, atol=1e-6)
        or
        np.isclose(coords.base[-1], total_tau + temp_res.base, atol=1e-6)
        or
        (total_tau <= coords.base[-1] <= (total_tau + temp_res.base))
        )


# >> Temp Filters

def test_tq_filt_args_array_round_trip():
    test_args = filters.do.TQTempFiltArgs(
            tau=Time(3), w=10, phi=100
        )


    assert np.all(test_args.array() == np.array([3, 10, 100]))


t = filters.do.TQTempFiltParams(
    amplitude=44,
    arguments=filters.do.TQTempFiltArgs(
        tau=Time(15), w=3, phi=0.3))


def test_tq_params_dict_conversion():
    putative_dict = {
        'amplitude': t.amplitude,
        'tau': asdict(t.arguments.tau),
        'w': t.arguments.w,
        'phi': t.arguments.phi
    }

    assert putative_dict == t.to_flat_dict()


def test_tq_params_array_round_trip():

    assert t == do.TQTempFiltParams.from_iter(t.array())



basic_float_strat = st.floats(min_value=1, max_value=10, allow_infinity=False, allow_nan=False)


# > Spatial Filters

# >> Spatial Coords Management

def test_basic_coord_res_exception():
    """Basic test of check_spat_ext_res function

    IE, does it pick up different units and extents that aren't multiples of res
    and raise the appropriate exception.
    """

    # check whole multiple exception
    ext = ArcLength(10)
    res = ArcLength(3)

    with raises(exc.CoordsValueError):
        ff.check_spat_ext_res(ext, res)

    # check unit exception
    ext = ArcLength(1, 'sec')
    res = ArcLength(1, 'mnt')

    with raises(exc.CoordsValueError):
        ff.check_spat_ext_res(ext, res)


arclength_units_strat = st.sampled_from(['mnt', 'sec', 'deg'])


@st.composite
def unrounded_coord_and_res(
    draw, min_value, max_value, max_coord_factor) -> Tuple[ArcLength[int], ArcLength[float]]:
    """Generate integer res and larger float coord each in random units

    res and coord are resolution and coordinate

    values are drawn from int or float strategies

    units are drawn from another composite: arclength_units_strat

    Args:
        min_value: min of res value
        max_value: max of res value
        max_coord_factor: max factor by which coord will be greater than res (min=2)

    Returns:
        res (ArcLength): resolution
        coord (ArcLength): coordinate
    """
    res_value: int = draw(st.integers(min_value=min_value, max_value=max_value))
    coord_factor: float = draw(  # factor for ... coord = res_value * coord_factor
        st.floats(min_value=2, max_value=max_coord_factor))
    coord_value = res_value * coord_factor

    # allow to have different base units
    res_unit = draw(arclength_units_strat)
    coord_unit = draw(arclength_units_strat)

    res = ArcLength(res_value, res_unit)
    # coord_value is in same unit as res_unit
    # Instantiate, then convert value to coord_unit, then
    # re-instantiate with coord_unit
    coord = ArcLength(coord_value, res_unit).as_new_unit(coord_unit)
    # coord = ArcLength(
    #     ArcLength(coord_value, res_unit)[coord_unit],
    #     coord_unit)

    return res, coord


@given(coord_res=unrounded_coord_and_res(1, 1000, 100))
def test_round_coord_to_res_multiple(
        coord_res: Tuple[ArcLength[int], ArcLength[float]]):
    """Are coords accurately snapped to whole integer multiple of resolution
    """

    res, coord = coord_res
    event(f'coord_unit: {coord.unit}, res_unit: {res.unit}')
    # event(f'coord_val: {coord.value}, res_val: {res.value}, units: {coord.unit}, {res.unit}')

    # test low and high work as expected ... simple
    new_coords = [
        ff.round_coord_to_res(coord, res, low=True),
        ff.round_coord_to_res(coord, res),
        ff.round_coord_to_res(coord, res, high=True)
    ]

    # low and high flags produce correctly higher or lower values
    assert sorted(new_coords, key=lambda c: c.value) == new_coords

    # test that coords are snapped to res
    assert all([
        (nc.value % res.value) == 0
        for nc in new_coords
    ])


@st.composite
def not_whole_number_multiple_arclengths(
        draw, min_value, max_value):
    """Strategy for arclengths where one is not whole number multiple of other

    Returns lower, main ... main is not whole multiple of lower
    """

    main_val = draw(st.integers(
        min_value=min_value, max_value=max_value
        ))
    lower_val = draw(st.integers(
        min_value=min_value, max_value=max_value
        ))

    main_unit = draw(arclength_units_strat)
    lower_unit = draw(arclength_units_strat)

    main = ArcLength(main_val, main_unit)
    lower = ArcLength(lower_val, lower_unit)

    assume(not (main.in_same_units_as(lower).value % lower.value == 0))

    return lower, main


@given(
    coord_args=not_whole_number_multiple_arclengths(min_value=1, max_value=10000))
def test_coord_res_exception(coord_args):
    """Exception should be raised when spatial extent not a whole multiple of resolution
    """

    spat_res, spat_ext = coord_args

    spat_ext = spat_ext.in_same_units_as(spat_res)

    event(f'res: {spat_res}, ext')

    with raises(exc.CoordsValueError):
        ff.check_spat_ext_res(spat_ext, spat_res)


@given(
    spat_ext=st.floats(
        min_value=1, max_value=1000, allow_infinity=False, allow_nan=False))
def test_coords_center_zero_basic(spat_ext: float):
    """Test that at coords.size//2 is always the coordinate value 0

    Uses default value of spat_res: 1 mnt ... simple version of the test
    """

    spat_extent = ArcLength(spat_ext, 'mnt')
    coords = ff.mk_spat_coords_1d(spat_ext=spat_extent)

    assert np.isclose(coords.mnt[coords.base.size//2], 0.0)  # type: ignore


@given(coord_args=unrounded_coord_and_res(
        min_value=1, max_value=1000, max_coord_factor=100))
def test_coords_center_zero_symmetry_res_snapped_multi_res(coord_args):
    """Test that coords at center are zero like above but with variable resolution

    Also, symmetrical and snapped to res ...
    ... basically the fundamental guarantees for spatial coords

    Not quite a pure unit test here ... but they all kinda depend on each other
    in the end I guess??
    """

    res_arclength, ext_arclength = coord_args

    # what combinations of units are used by the strategy?
    # event(f'res_unit: {res_arclength}, ext_unit: {ext_arclength}')
    # event(f'res_unit: {res_arclength.unit}, ext_unit: {ext_arclength.unit}')

    coords = ff.mk_spat_coords_1d(res_arclength, ext_arclength)

    ends = coords.value[0], coords.value[-1]

    # test that symmetrical
    assert np.isclose(ends[0], -1 * ends[1])

    # test snapped to res
    # coords are already in same unit as res
    assert (ends[0] % res_arclength.value) == 0
    assert (ends[1] % res_arclength.value) == 0

    # do unit conversion but rely only on being close
    # calculate proportion of a res value away from res
    # (should be floating point error difference only)
    # mod 1 to get only decimal portion of floating point
    # this represents how many res values the coord is from a whole integer
    distance = (abs(coords[res_arclength.unit][0]) / res_arclength.value) % 1
    # want smallest distance between either 0 or 1
    corrected_distance = distance if (distance<0.5) else abs(1-distance)
    np.isclose(corrected_distance, 0)

    # assert np.isclose((abs(coords[res_arclength.unit][0]) % res_arclength.value),  0)

    # test shape is odd
    assert (coords.value.size % 2) == 1
    # test that center is zero
    assert np.isclose(
        coords.value[coords.value.size//2],
        0)  # type: ignore


@given(coord_args=unrounded_coord_and_res(
        min_value=1, max_value=1000, max_coord_factor=100))
def test_spat_coords_int(coord_args: Tuple[ArcLength[int], ArcLength[float]]):
    """spat coords array is integer not float
    which sould be the case from all the rounding and snapping to the res arg

    This may not always be the case ... the code base may change.
    """

    res_arclength, ext_arclength = coord_args
    coords = ff.mk_spat_coords_1d(res_arclength, ext_arclength)

    assert np.issubdtype(coords.value.dtype, np.integer)


def test_spat_coords_origin_bottom_left():
    r,e = ArcLength(1, 'mnt'), ArcLength(500, 'mnt')
    X,Y = ff.mk_spat_coords(r, e)

    assert X.value[0, 0] < X.value[0, -1]  # X goes upward left to right
    assert Y.value[0, 0] > Y.value[-1, 0]  # Y goes upward bottom to top


@given(
    spat_res=st.integers(1, 10),
    spat_ext_factor=st.floats(10, 100, allow_nan=False, allow_infinity=False),
    )
def test_spat_coords_match_1d_coords(
        spat_res, spat_ext_factor):

    spat_ext = spat_res * spat_ext_factor

    x, y = ff.mk_spat_coords(
            spat_res=ArcLength(spat_res, 'mnt'),
            spat_ext=ArcLength(spat_ext, 'mnt'),
        )

    assert x.base.shape == y.base.shape  # basic test, mostly trivial

    # spatial slice
    coord_1d = ff.mk_spat_coords_1d(ArcLength(spat_res, 'mnt'), ArcLength(spat_ext, 'mnt'))

    assert np.allclose(coord_1d.base, x.base[0,:])  # first dimension is Y axis, second x axis.
    assert coord_1d.base.shape[0] == x.base.shape[1]  # lengths should match


@mark.proto
@given(
    spat_res=st.integers(1, 10),
    spat_ext_factor=st.floats(10, 100, allow_nan=False, allow_infinity=False),
    )
def test_spat_coords_match_sd_limited_1d_coords(
        spat_res, spat_ext_factor):

    spat_ext = spat_res * spat_ext_factor

    x, y = ff.mk_spat_coords(
            spat_res=ArcLength(spat_res, 'mnt'),
            spat_ext=ArcLength(spat_ext, 'mnt'),
        )

    assert x.base.shape == y.base.shape  # basic test, mostly trivial

    # spatial slice
    coord_1d = ff.mk_sd_limited_spat_coords(ArcLength(spat_res, 'mnt'), ArcLength(spat_ext, 'mnt'))

    assert np.allclose(coord_1d.base, x.base[0,:])  # first dimension is Y axis, second x axis.
    assert coord_1d.base.shape[0] == x.base.shape[1]  # lengths should match


@mark.proto
@given(
    spat_res=st.integers(1, 10),
    sf_max_sd=st.floats(5, 100, allow_nan=False, allow_infinity=False),
    )
def test_spat_coords_span_sufficient_sd_multiples(
        spat_res, sf_max_sd):

    coords = ff.mk_sd_limited_spat_coords(
            spat_res=ArcLength(spat_res, 'mnt'),
            sd=ArcLength(sf_max_sd, 'mnt'),
        )

    # times 2 as it is a radial factor ... diameter (ie, full size of coords) is double
    minimum_sd_multiples = settings.SimulationParams.spat_filt_sd_factor * 2
    actual_sd_multiples = ((coords.mnt[-1] - coords.mnt[0]) / sf_max_sd)

    # event(f'n_multiples: {actual_sd_multiples}')

    assert  actual_sd_multiples > minimum_sd_multiples


@mark.proto
@given(
    spat_res=st.integers(1, 10),
    sf_max_sd=st.floats(5, 100, allow_nan=False, allow_infinity=False),
    sd_unit=st.sampled_from(['mnt', 'deg', 'sec'])
    )
def test_spat_coords_span_extent_matches_analytical_value(
        spat_res, sf_max_sd, sd_unit):
    """Just to ensure that the spatial extent of a rendered spatial filter
    can be accurately determined from the spatial filter's parameters
    """
    spat_res = ArcLength(spat_res, 'mnt')
    sd = ArcLength(sf_max_sd, 'mnt').as_new_unit(sd_unit)
    coords = ff.mk_sd_limited_spat_coords(
                spat_res=spat_res,
                sd=sd,
            )
    coords_size = coords.value.size
    predicted_coords_radius = (
        ff.mk_rounded_spat_radius(
            spat_res=spat_res,
            spat_ext=ff.mk_spat_ext_from_sd_limit(sd)
            ).value
        )
    predicted_coords_size = ((predicted_coords_radius / spat_res.value) * 2) + 1

    assert coords_size == predicted_coords_size

    # more manual calculation ... just to make sure
    # multiply the sd by the sd_factor in settings
    predicted_sd_limit = ArcLength(
            sd.value * 2*settings.simulation_params.spat_filt_sd_factor,
            sd.unit
            )
    # make a rounded spat_radius, double, divide by size of spat_res and add 1 for zero-center
    predicted_coords_radius2 = (
            (
                ff.mk_rounded_spat_radius(spat_res, predicted_sd_limit).value
                * 2
                / spat_res.value
            )
            + 1
        )

    assert coords_size == predicted_coords_radius2


# >>> spat filt sd strats
cent_sd_strat = st.floats(min_value=10, max_value=29, allow_infinity=False, allow_nan=False)
surr_sd_strat = st.floats(min_value=30, max_value=50, allow_infinity=False, allow_nan=False)

@mark.integration
# @hypothesis.settings(deadline=300)  # deadline of 300 ms from default of 200
@given(
        spat_res=st.integers(1, 10),
        spat_res_unit=st.sampled_from(['mnt', 'sec']),  # deg would be too big to be integer
        cent_h_sd=cent_sd_strat, cent_v_sd=cent_sd_strat, # though same start, diff vals
        surr_h_sd=surr_sd_strat, surr_v_sd=surr_sd_strat,
        rf_unit=st.sampled_from(['mnt', 'deg', 'sec'])
    )
def test_spat_filt_size_prediction(
        spat_res, spat_res_unit,
        cent_h_sd, cent_v_sd,
        surr_h_sd, surr_v_sd,
        rf_unit):
    """Test accurate prediction of rendered spat filt for various combinations of parameters

    Including different units for rf and spat_res, within reason and small enough to maintain
    workable array sizes
    """

    spat_res = ArcLength(
            int(ArcLength(spat_res, 'mnt')[spat_res_unit]),
            spat_res_unit
        )
    # event(f'{spat_res, cent_h_sd, cent_v_sd, surr_h_sd, surr_v_sd, rf_unit}')

    # ensure values are kept small enough to not blow up
    # the size of arrays
    if rf_unit == 'deg':
        cent_h_sd, cent_v_sd, surr_h_sd, surr_v_sd = (
            v / 55  # not 60, but 55, to create more random vals
                for v
                in (cent_h_sd, cent_v_sd, surr_h_sd, surr_v_sd )
            )
    if rf_unit == 'sec':
        cent_h_sd, cent_v_sd, surr_h_sd, surr_v_sd = (
            v * 65  # not 60, but 65, to create more random vals
                for v
                in (cent_h_sd, cent_v_sd, surr_h_sd, surr_v_sd )
            )

    dog_rf_params = do.DOGSpatFiltArgs(
        cent=do.Gauss2DSpatFiltParams.from_iter(
            [1, cent_h_sd, cent_v_sd], arclength_unit=rf_unit),
        surr=do.Gauss2DSpatFiltParams.from_iter(
            [1, surr_h_sd, surr_v_sd], arclength_unit=rf_unit)
        )

    xc, yc = ff.mk_spat_coords(
        sd=dog_rf_params.max_sd(),
        spat_res=spat_res)

    spat_filt_rendered = ff.mk_dog_sf(xc, yc, dog_rf_params)

    predicted = ff.spat_filt_size_in_res_units(
                    spat_res=spat_res,
                    sf=dog_rf_params)

    assert xc.value.shape[0] == spat_filt_rendered.shape[0] == predicted


@given(spat_res_unit=st.sampled_from(['sec','mnt', 'deg']))
def test_spat_coords_units_correct(spat_res_unit):

    spat_res = ArcLength(1, spat_res_unit)
    spat_ext = ArcLength(spat_res.deg * 10, 'deg')  # ensure always bigger

    x,y = ff.mk_spat_coords(spat_res, spat_ext)

    assert x.unit == y.unit == spat_res_unit


# >> Spatio-Temp Coords

@given(
    spat_res=st.integers(1, 10),
    temp_res=st.floats(0.01, 0.1, allow_nan=False, allow_infinity=False),
    spat_ext_factor=st.floats(10, 100, allow_nan=False, allow_infinity=False),
    temp_ext_factor=st.floats(10, 100, allow_nan=False, allow_infinity=False),
    )
def test_spat_temp_coords_match_1d_coords(
        spat_res, temp_res, spat_ext_factor, temp_ext_factor):

    spat_ext = spat_res * spat_ext_factor
    temp_ext = temp_res * temp_ext_factor

    x, y, t = ff.mk_spat_temp_coords(
            spat_res=ArcLength(spat_res, 'mnt'),
            temp_res=Time(temp_res, 'ms'),
            spat_ext=ArcLength(spat_ext, 'mnt'),
            temp_ext=Time(temp_ext, 'ms')
        )

    assert x.base.shape == y.base.shape == t.base.shape  # basic test, mostly trivial

    # spatial slice
    coord_1d = ff.mk_spat_coords_1d(ArcLength(spat_res, 'mnt'), ArcLength(spat_ext, 'mnt'))

    assert coord_1d.base.shape[0] == x.base.shape[1]  # lengths should match
    assert np.allclose(coord_1d.base, x.base[0,:,0])  # first dimension is Y axis, second, is X axis

    # temporal slice
    coord_1d = ff.mk_temp_coords(Time(temp_res, 'ms'), Time(temp_ext, 'ms'))

    assert coord_1d.base.shape[0] == x.base.shape[2]
    assert np.allclose(coord_1d.base, t.base[0,0,:])  # third dimension is third axis


@given(
    spat_res_unit=st.sampled_from(['sec','mnt', 'deg']),
    temp_res_unit=st.sampled_from(['us', 'ms', 's']))
def test_spat_temp_coords_units_correct(spat_res_unit, temp_res_unit):

    spat_res = ArcLength(1, spat_res_unit)
    temp_res = Time(1, temp_res_unit)
    spat_ext = ArcLength(spat_res.deg * 10, 'deg')  # ensure always bigger
    temp_ext = Time(temp_res.s * 10, 's')

    x,y,t = ff.mk_spat_temp_coords(spat_res, temp_res, spat_ext, temp_ext )

    assert x.unit == y.unit == spat_res_unit
    assert t.unit == temp_res_unit


# >> DoG and Gaussian Filters

@given(
    x_sd=basic_float_strat,
    y_sd=basic_float_strat,
    mag=basic_float_strat
    )
def test_gauss_2d_sum(x_sd: float, y_sd: float, mag: float):
    """Sum of gauss_2d is defined by magnitude

    Generate gauss_2d from ff.mk_gauss_2d and check that integral
    (sum * resolution) is the same as the magnitude (ie, integral==1)
    """

    # factor by which to make RF bigger than x_sd/y_sd args
    # to ensure integer spat res but sufficient RF and extent size
    rf_factor = 10

    gauss_params = do.Gauss2DSpatFiltParams(
        amplitude=mag,
        arguments=do.Gauss2DSpatFiltArgs(
                # to ensure spatial resolution is high enough
                # multiple sd by rf_factor, as has min 1, and res must be an integer
                ArcLength(x_sd * rf_factor, 'mnt'),
                ArcLength(y_sd * rf_factor, 'mnt')
            )
        )

    # ensure resolution high enough for sd
    spat_res = ArcLength(
        (int(min(x_sd, y_sd))),
        'mnt'
        )
    # ensure extent high enough for capture full gaussian
    spat_ext = ArcLength(
        int(
            2 *
            5*np.ceil(
                    # multiply rf_factor to ensure extent is sufficient
                    max(x_sd, y_sd) * rf_factor
                )
            + 1
            ),
        'mnt'
        )

    xc, yc = ff.mk_spat_coords(spat_res=spat_res, spat_ext=spat_ext)

    gauss_2d = ff.mk_gauss_2d(xc, yc, gauss_params=gauss_params)

    assert np.isclose(gauss_2d.sum()*spat_res.mnt**2, mag)  # type: ignore


@hypothesis.settings(deadline=300)  # deadline of 300 ms from default of 200
@given(
    mag_cent=basic_float_strat,
    mag_surr=basic_float_strat,
    cent_h_sd=basic_float_strat,
    cent_v_sd=basic_float_strat,
    surr_h_sd=basic_float_strat,
    surr_v_sd=basic_float_strat
    )
def test_dog_rf(
        cent_h_sd: float, cent_v_sd: float,
        surr_h_sd: float, surr_v_sd: float,
        mag_cent: float, mag_surr: float):
    """mk_dog_rf produces same output as manual 2d rf calc

    Significance being that mk_dog_rf treats 1d gaussians as separable
    """

    ####
    # all sd vals are presumed in minutes

    # factor by which to make RF bigger than x_sd/y_sd args
    # to ensure integer spat res but sufficient RF and extent size
    rf_factor = 10

    # preparing params and coords
    dog_rf_params = do.DOGSpatFiltArgs(
        cent=do.Gauss2DSpatFiltParams.from_iter(
            [mag_cent, cent_h_sd * rf_factor, cent_v_sd * rf_factor],
            arclength_unit='mnt'),
        surr=do.Gauss2DSpatFiltParams.from_iter(
            [mag_surr, surr_h_sd * rf_factor, surr_v_sd * rf_factor],
            arclength_unit='mnt')
        )

    # ensure resolution high enough for sd
    spat_res = (int(min(
        [cent_h_sd, cent_v_sd, surr_h_sd, surr_v_sd]
        )))
    # ensure extent high enough for capture full gaussian
    spat_ext = int(
        2 *
        5*np.ceil(
            max([cent_h_sd, cent_v_sd, surr_h_sd, surr_v_sd]) * rf_factor
            )
        + 1)

    xc: ArcLength
    yc: ArcLength
    xc, yc = ff.mk_spat_coords(spat_res=ArcLength(spat_res), spat_ext=ArcLength(spat_ext))

    ###############
    # making dog rf
    dog_rf = ff.mk_dog_sf(xc, yc, dog_args=dog_rf_params)

    ###############
    # making rf with direct code

    # scale gauss parameters as used above
    (cent_h_sd, cent_v_sd, surr_h_sd, surr_v_sd) = (
        cent_h_sd * rf_factor,
        cent_v_sd * rf_factor,
        surr_h_sd * rf_factor,
        surr_v_sd * rf_factor)

    rf_cent = (
        # mag divide by normalising factor with both sds (equivalent to sq if they were identical)
        (mag_cent / (2 * np.pi * cent_v_sd * cent_h_sd)) *
        np.exp(
            - (
                (xc.mnt**2 / (2 * cent_h_sd**2)) +
                (yc.mnt**2 / (2 * cent_v_sd**2))
            )
        )
    )
    rf_surr = (
        (mag_surr / (2 * np.pi * surr_v_sd * surr_h_sd)) *
        np.exp(
            - (
                (xc.mnt**2 / (2 * surr_h_sd**2)) +
                (yc.mnt**2 / (2 * surr_v_sd**2))
            )
        )
    )

    rf = rf_cent - rf_surr

    assert np.allclose(rf, dog_rf)  # type: ignore


# >> Fourier

@mark.parametrize(
    'freqs, factor',
    [
        (0, 1),
        (np.array([1, 0, 12.5, -3, -0]), np.array([2, 1, 2, 2, 1]))
    ],
    ids=['floats', 'array']
    )
def test_collapse_symmetry_fact(freqs, factor):

    assert np.all(ff._mk_ft_freq_symmetry_factor(freqs) == factor)


@mark.parametrize(
    'xfreqs, yfreqs, factor',
    [
        (np.array([1, 0, 12.5, -3, -0]), np.array([1, 0, 0, 4, 0]), np.array([2, 1, 2, 2, 1]))
    ]
    )
def test_collapse_symmetry_fact_2d(xfreqs, yfreqs, factor):

    assert np.all(ff._mk_ft_freq_symmetry_factor_2d(xfreqs, yfreqs) == factor)


@mark.parametrize(
    'freqs, factor',
    [
        (0, 1),
        (np.array([1, 0, 12.5, -3, -0, 0.0]), np.array([2, 1, 2, 2, 1, 1]))
    ],
    ids=['floats', 'array']
    )
def test_gauss_1d_ft_symmetry(freqs, factor):

    freqs = SpatFrequency(freqs)

    assert np.all(
        ff.mk_gauss_1d_ft(freqs, collapse_symmetry=True) == factor * ff.mk_gauss_1d_ft(freqs)
        )


@mark.parametrize(
    'xfreqs, yfreqs, factor',
    [
        (np.array([1, 0, 12.5, -3, -0]), np.array([1, 0, 0, 4, 0]), np.array([2, 1, 2, 2, 1]))
    ]
    )
def test_gauss_2d_ft_symmetry(xfreqs, yfreqs, factor):

    xfreqs, yfreqs = SpatFrequency(xfreqs), SpatFrequency(yfreqs)

    params = do.Gauss2DSpatFiltParams(
        amplitude=1, arguments=do.Gauss2DSpatFiltArgs(h_sd=ArcLength(10), v_sd=ArcLength(10))
        )

    assert np.all(
        ff.mk_gauss_2d_ft(xfreqs, yfreqs, gauss_params=params, collapse_symmetry=True) ==
        factor * ff.mk_gauss_2d_ft(xfreqs, yfreqs, gauss_params=params)
        )


@mark.parametrize(
    'xfreqs, yfreqs, factor',
    [
        (
            np.array([1, 0, 12.5, -3, -0]),
            np.array([1, 0, 0,     4,  0]),
            np.array([2, 1, 2,     2,  1]))
    ]
    )
def test_dog_sf_ft_symmetry(xfreqs, yfreqs, factor):

    xfreqs, yfreqs = SpatFrequency(xfreqs), SpatFrequency(yfreqs)

    dog_rf_params = do.DOGSpatFiltArgs(
        cent=do.Gauss2DSpatFiltParams.from_iter(
            [1.1, 10, 10], arclength_unit='mnt'),
        surr=do.Gauss2DSpatFiltParams.from_iter(
            [0.9, 30, 30], arclength_unit='mnt')
        )

    assert np.all(
        ff.mk_dog_sf_ft(xfreqs, yfreqs, dog_rf_params, collapse_symmetry=True) ==
        factor * ff.mk_dog_sf_ft(xfreqs, yfreqs, dog_rf_params, collapse_symmetry=False)
        )

@given(sd_val=st.floats(min_value=10, max_value=100))
def test_gauss_1d_ft(sd_val: float):
    """manual fft and custom fourier for 1d gauss match
    """

    sd = ArcLength(sd_val, 'mnt')
    spat_res = ArcLength(1, 'mnt')
    coords = ff.mk_sd_limited_spat_coords(spat_res = spat_res, sd=sd)
    t_gauss = ff.mk_gauss_1d(coords=coords, sd=sd)

    assert np.isclose(t_gauss.sum(), 1)  # integral should always be 1

    # use numpy fft
    tg_fft = np.abs(fft.rfft(t_gauss))
    tg_fft_freq = fft.rfftfreq(coords.mnt.shape[0], d=spat_res.mnt)
    # use this codebase's function
    freq = SpatFrequency(tg_fft_freq, 'cpm')
    tg_ft = ff.mk_gauss_1d_ft(freqs=freq, sd=sd)

    # always some errors in comparing analytical and numerical fouriers ...
    # just ease the # tolerance a little
    # (maybe should use relative tolerance instead?)
    assert np.all(np.isclose(tg_fft, tg_ft, atol=1e-5))


# Do I need to use hypothesis for this?  Main point is that any equivalence is good?
# @mark.parametrize(
#         'cent_h_sd,cent_v_sd,surr_h_sd,surr_v_sd,mag_cent,mag_surr',
#         [(10, 13, 30, 30, 17/16, 15/16)]
#     )

@hypothesis.settings(deadline=300)  # deadline of 300 ms from default of 200
@given(
        cent_h_sd=cent_sd_strat, cent_v_sd=cent_sd_strat,
        surr_h_sd=surr_sd_strat, surr_v_sd=surr_sd_strat,
        mag_cent=st.floats(min_value=1.1, max_value=5, allow_infinity=False, allow_nan=False),
        mag_surr=st.floats(min_value=0.5, max_value=1, allow_infinity=False, allow_nan=False)
    )
def test_dog_rf_gauss2d_fft(
        cent_h_sd: float, cent_v_sd: float,
        surr_h_sd: float, surr_v_sd: float,
        mag_cent: float, mag_surr: float):
    """mk_dog_rf_ft produces same output as manual fft

    Significance being that x and y separability are used by mk_dog_rf_ft to
    produce a 2d ft from 1d fts
    """

    dog_rf_params = do.DOGSpatFiltArgs(
        cent=do.Gauss2DSpatFiltParams.from_iter(
            [mag_cent, cent_h_sd, cent_v_sd], arclength_unit='mnt'),
        surr=do.Gauss2DSpatFiltParams.from_iter(
            [mag_surr, surr_h_sd, surr_v_sd], arclength_unit='mnt')
        )

    # # [><] WARNING [><] ##
    # Note spat_res and spat_ext in 'mnt' units
    # MUST MATCH with 'cpm' in spatfrequency objects below

    # ensure resolution high enough for sd
    # thus ... divide by 10?
    # spat_res = ArcLength(
    #     (np.floor(np.min(np.array([cent_h_sd, cent_v_sd, surr_h_sd, surr_v_sd])))) / 10, 'mnt')
    # fix resolution at 1 mnt so that tests pass
    # bigger res values lead to more fuzziness in the FFT and errors in this test
    spat_res = ArcLength(1, 'mnt')
    # ensure extent high enough for capture full gaussian
    # thus ... add 1?
    spat_ext = ArcLength(
        2 * 5*np.ceil(np.max(np.array([cent_h_sd, cent_v_sd, surr_h_sd, surr_v_sd]))) + 1, 'mnt')

    xc, yc = ff.mk_spat_coords(spat_res=spat_res, spat_ext=spat_ext)

    dog_rf = ff.mk_dog_sf(xc, yc, dog_args=dog_rf_params)

    dog_rf_fft = np.abs(fft.fftshift(fft.fft2(dog_rf)))
    # using spat_res.value so that just have to match units above and below (mnt <-> cpm)
    dog_rf_fft_freqs = fft.fftshift(fft.fftfreq(xc.mnt.shape[0], d=spat_res.value))  # type: ignore

    # use 'cpm' as used 'mnt' above
    fx, fy = (
        SpatFrequency(f, unit='cpm') for f
        in np.meshgrid(dog_rf_fft_freqs, dog_rf_fft_freqs)  # type: ignore
        )
    # FT = T * FFT ~ amplitude of convolution (T -> spatial resolution)
    dog_rf_ft = (
        ff.mk_dog_sf_ft(fx, fy, dog_args=dog_rf_params, collapse_symmetry=False)
        /
        spat_res.value)

    # atol adjusted as there's some systemic error in comparing continuous with FFT DFT
    # check graphs of the difference to see

    # hopefully this isn't too much of a problem!
    # It's pretty forgiving though ... doubling the ft didn't even cause a problem

    is_close = np.isclose(dog_rf_ft, dog_rf_fft, atol=1e-5)  # type: ignore
    # if np.any(~is_close):
    #     diffs = np.abs(
    #         dog_rf_ft[~is_close] - dog_rf_fft[~is_close]
    #         ) / dog_rf_ft[~is_close]  # type: ignore
    #     assert diffs.mean() < 0.05
    assert np.all(is_close)
    # assert (is_close.mean() > 0.95)  # type: ignore


## Careful!!  Can easily take up lots of memory with large arrays

@given(
    mag_cent=basic_float_strat,
    mag_surr=basic_float_strat,
    cent_h_sd=basic_float_strat,
    # cent_v_sd=basic_float_strat,
    surr_h_sd=basic_float_strat,
    # surr_v_sd=basic_float_strat
    )
def test_dog_ft_2d_to_1d(
        cent_h_sd: float, surr_h_sd: float,
        mag_cent: float, mag_surr: float):
    """Compare simple 1d continuous fourier with slice of 2d

    If this passes, then direct mapping from 2d to 1d fourer transform for gaussians

    1d params:cent_amp, surr_amp, cent_sd, surr_sd
    -> 2d params: h_sd, v_sd = sd (as radially symmetrical)

    Works because if all y frequencies (fy) are zero (because of the slice) then this term
    in the equation has no impact
    """
    # breakpoint()
    # expand size of rf so that spat_res can be integer but 10 times smaller
    rf_factor = 10
    # these have values 10X the args
    cent_h_sd_al, surr_h_sd_al = (
        ArcLength(sd * rf_factor, 'mnt') for sd in (cent_h_sd, surr_h_sd))

    dog_sf_args = do.DOGSpatFiltArgs(
        cent=do.Gauss2DSpatFiltParams(
            amplitude=mag_cent,
            arguments=do.Gauss2DSpatFiltArgs(h_sd=cent_h_sd_al, v_sd=cent_h_sd_al)
            ),
        surr=do.Gauss2DSpatFiltParams(
            amplitude=mag_surr,
            arguments=do.Gauss2DSpatFiltArgs(h_sd=surr_h_sd_al, v_sd=surr_h_sd_al)
            )
        )

    # ensure resolution high enough for sd
    # value uses original args (not 10X)
    spat_res = ArcLength(
        int(min([cent_h_sd, surr_h_sd])),
        'mnt')
    # ensure extent high enough for capture full gaussian
    # values use the arclengths with 10X value of original args
    spat_ext: ArcLength[float] = ArcLength(
        int(
            2 *
            5*np.ceil(max([cent_h_sd_al.mnt, surr_h_sd_al.mnt])) + 1
            ),
        'mnt')

    # spat_res, spat_ext = 1, 300
    coords = ff.mk_spat_coords_1d(spat_res=spat_res, spat_ext=spat_ext)
    ft_freq = fft.fftshift(fft.fftfreq(coords.mnt.size, d=spat_res.value))  # type: ignore
    fx, fy = (
        SpatFrequency(f) for f 
        in np.meshgrid(ft_freq, ft_freq))  # type: ignore

    # 2d ft
    dog_ft = ff.mk_dog_sf_ft(fx, fy, dog_args=dog_sf_args, collapse_symmetry=False)
    # take center slice
    x_cent_dog_ft = dog_ft[int(coords.mnt.size//2), :]

    # make 1d for cent and surr
    cent_ft = ff.mk_gauss_1d_ft(
        SpatFrequency(ft_freq),
        amplitude=dog_sf_args.cent.amplitude, sd=dog_sf_args.cent.arguments.h_sd,
        collapse_symmetry=False)
    surr_ft = ff.mk_gauss_1d_ft(
        SpatFrequency(ft_freq),
        amplitude=dog_sf_args.surr.amplitude, sd=dog_sf_args.surr.arguments.h_sd,
        collapse_symmetry=False)

    # make 1d dog ft (by subtracting)
    dog_1d_ft = cent_ft - surr_ft

    assert np.allclose(x_cent_dog_ft, dog_1d_ft)  # type: ignore


def test_sf_ft_positive_for_negative_freq():
    """Ensure that spatial ft function is radially invariant (positive for neg freqs)

    In many ways a test of the specific mathematics of the FT function for DOG sf
    """

    #  Make radially symmetric DOG filter
    sf = do.DOGSpatFiltArgs(
        cent=do.Gauss2DSpatFiltParams(
            amplitude=36.,
            arguments=do.Gauss2DSpatFiltArgs(
                h_sd=ArcLength(value=1., unit='mnt'),
                v_sd=ArcLength(value=1., unit='mnt'))
            ),
        surr=do.Gauss2DSpatFiltParams(
            amplitude=21.,
            arguments=do.Gauss2DSpatFiltArgs(
                h_sd=ArcLength(value=6., unit='mnt'),
                v_sd=ArcLength(value=6., unit='mnt')))
        )

    oris = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    spat_freq = SpatFrequency(2)
    ft_vals = []

    for o in oris:
        freq_x, freq_y = ff.mk_sf_ft_polar_freqs(ArcLength(o, 'deg'), spat_freq)
        ft_val = ff.mk_dog_sf_ft(freq_x, freq_y, sf)
        ft_vals.append(ft_val)

    assert all([
            np.isclose(ft_val, ft_vals[0])
            for ft_val in ft_vals
        ])

# >> Spatial filter rotation
@mark.proto
def test_sf_rotation_does_not_change_array_size():

    dog_rf_params = do.DOGSpatFiltArgs(
        cent=do.Gauss2DSpatFiltParams.from_iter(
            [1.1, 10, 10], arclength_unit='mnt'),
        surr=do.Gauss2DSpatFiltParams.from_iter(
            [0.9, 30, 30], arclength_unit='mnt')
        )

    xc, yc = ff.mk_spat_coords(spat_res=ArcLength(1,'mnt'), spat_ext=ArcLength(100,'mnt'))
    dog_rf = ff.mk_dog_sf(xc, yc, dog_args=dog_rf_params)


    orientation_size_checks = [
        (ff.mk_oriented_sf(dog_rf, ArcLength(ori, 'deg')).shape == dog_rf.shape)
        for ori in np.linspace(0, 180, 10)
    ]

    assert all(orientation_size_checks)


def test_sf_rotation_produces_correct_orientation_preference():
    assert False

# > Convolution management and adjustment

# limits on sd: keep res at 1.mnt, and don't consume too much memory
@given(
    sd=st.floats(min_value=10, max_value=1000, allow_infinity=False, allow_nan=False)
    )
def test_sf_conv_amp_1d(sd):
    "Convolution amplitude estimation for 1d convolution of sinusoid"

    sd = ArcLength(sd, 'mnt')
    # one cycle per 3 sd for signal ... ensures amplitude of convolution always substantial
    signal_freq = SpatFrequency(1/(3*sd.base))
    spat_ext: ArcLength[float] = ArcLength(int(20*sd.base))

    # make gaussian filter
    coords = ff.mk_spat_coords_1d(spat_ext=spat_ext)
    gauss = ff.mk_gauss_1d(coords=coords, sd=sd)

    signal = np.cos(signal_freq.cpd_w * coords.deg)  # should produce pure radians (rad/deg * deg)

    conv_signal = convolve(gauss, signal, mode='same')

    # Find amplitude of result of convolution
    mid_point = signal.size // 2  # mid point here guaranteed by mk_spat_coord

    # get amplitude from middle peak at 0.0 (as cosine) and next trough
    max_val = conv_signal[mid_point]

    trough_idxs = argrelmin(conv_signal[mid_point:])
    trough_idx = trough_idxs[0][0]  # get first minima from first array of indices
    min_val = conv_signal[mid_point+trough_idx]

    # Amplitude of sinusoid is from middle to peak (not peak to peak)
    actual_amplitude: float = (max_val - min_val) / 2

    conv_amp_est = ff.mk_gauss_1d_ft(signal_freq, amplitude=1, sd=sd)

    assert np.isclose(actual_amplitude, conv_amp_est, rtol=1e-2)  # type: ignore


# > test sf conv amp 2d

@mark.proto
@given(
    t_freq=st.one_of([
        # 0 or 0.5+, so that actual value of convolved signal easily isolated
        # temp_ext is 1s, so any freq >=0.5 for a sin wave should the peak nicely in the
        # middle
        st.just(0),
        st.floats(min_value=0.5, max_value=50, allow_infinity=False, allow_nan=False)
        ])
    )
def test_tq_tf_conv_amp_est(t_freq: float):
    "Test that estimate of tq tf accurate"

    temp_freq = TempFrequency(t_freq)

    temp_res = Time(value=0.1, unit='ms')  # high-ish resolution for accuracy
    temp_ext = Time(1.0, 's')

    temp_coords = ff.mk_temp_coords(temp_res, temp_ext)
    # use cos for 0hz and sin for else, to help isolate magnitude
    # for 0hz, cos is more accurate with amplitude of 1 (same as stimulus)
    sinusoid = np.cos if t_freq == 0 else np.sin
    signal = sinusoid(temp_freq.w * temp_coords.s)

    # ordinary tq temp filter
    tq_tf = do.TQTempFiltParams(
        amplitude=300,
        arguments=do.TQTempFiltArgs(
            tau=Time(value=14.0, unit='ms'), w=10, phi=1.12)
        )
    temp_filter = ff.mk_tq_tf(t=temp_coords, tf_params=tq_tf)

    # convolve and take only first part corresponding to filter size
    conv_signal = convolve(signal, temp_filter, mode='full')[:temp_coords.base.shape[0]]

    # 260 ms is approximately when this temp filter goes to zero
    inner_idx = int(temp_coords.base.shape[0] * 0.26)
    # inner_conv_signal = conv_signal[inner_idx:-inner_idx]
    inner_conv_signal = conv_signal[inner_idx:]

    # if DC stays zero (as DC of original signal is zero), then amplitude is simply max
    # as this is the magnitude from zero to positive peak
    # can therefore use either min or max
    # as trying to be most accurate, taking min from sinusoid is more likely to be accurate
    # as it will be later in the convoluved signal, and free of transient effects
    conv_amp = np.max(np.abs([inner_conv_signal.max(), inner_conv_signal.min()]))  # type: ignore
    # conv_amp = inner_conv_signal.max()
    est_conv_amp = ff.mk_tq_tf_conv_amp(temp_freq, tq_tf, temp_res)

    # print(temp_freq, conv_amp, est_conv_amp)

    assert np.isclose(conv_amp, est_conv_amp, rtol=1e-4)  # type: ignore


# make sf
# make sinusoid stim
    # Should make nice function for this
# simple convolution
# derive amplitude of convolution
# check against sf_ft

# > test fit spat filt

# > test spat filt fit is decent fit

def test_anisotropic_dog_rf_ft():

    # vertically elongated RF
    dog_args = do.DOGSpatFiltArgs(
        #                                            h   v
        cent=do.Gauss2DSpatFiltParams.from_iter([1, 10, 15], arclength_unit='mnt'),
        surr=do.Gauss2DSpatFiltParams.from_iter([1, 32, 32], arclength_unit='mnt')
        )

    # horizontal modulation (vertical gratings) ... should have greater amplitude
    theta = ArcLength(0.0)
    ft_amp_0 = ff.mk_dog_sf_ft(*ff.mk_sf_ft_polar_freqs(theta, SpatFrequency(1)), dog_args)

    # vertical modulation (horizontal gratings)
    theta = ArcLength(90.0)
    ft_amp_90 = ff.mk_dog_sf_ft(*ff.mk_sf_ft_polar_freqs(theta, SpatFrequency(1)), dog_args)

    assert ft_amp_0 > ft_amp_90

# test joint amp with manual params produces output

def test_joint_amp():

    # creating mock parameters from known parameters that work
    test_sf_params = do.DOGSpatFiltArgs(
        cent=do.Gauss2DSpatFiltParams(
            amplitude=36.4265938532914,
            arguments=do.Gauss2DSpatFiltArgs(
                h_sd=ArcLength(value=1.4763319256270793, unit='mnt'),
                v_sd=ArcLength(value=1.4763319256270793, unit='mnt'))),
        surr=do.Gauss2DSpatFiltParams(
            amplitude=21.123002637615535,
            arguments=do.Gauss2DSpatFiltArgs(
                h_sd=ArcLength(value=6.455530597672735, unit='mnt'),
                v_sd=ArcLength(value=6.455530597672735, unit='mnt')
                )
            )
        )
    test_tf_params = do.TQTempFiltParams(
        amplitude=301.92022743003315,
        arguments=do.TQTempFiltArgs(
            tau=Time(value=14.90265388080688, unit='ms'),
            w=11.420030523751624, phi=1.1201854280821189)
        )

    test_sf = do.DOGSpatialFilter(
        source_data=do.SpatFiltParams(
            data=None,  # type: ignore
            resp_params=do.SFRespMetaData(dc=15, tf=TempFrequency(value=4, unit='hz'),
                                          mean_lum=100, contrast=0.5),
            meta_data=None),
        parameters=test_sf_params,
        optimisation_result=None,  # type: ignore
        ori_bias_params=None)  # type: ignore

    test_tf = do.TQTempFilter(
        source_data=do.TempFiltParams(
            data=None,  # type: ignore
            resp_params=do.TFRespMetaData(
                dc=12, sf=SpatFrequency(value=0.8, unit='cpd'), mean_lum=100, contrast=0.4),
            meta_data=None
            ),
        parameters=test_tf_params,
        optimisation_result=None)  # type: ignore

    t_freq = TempFrequency(6.9)
    s_freq_x = SpatFrequency(3)
    s_freq_y = SpatFrequency(0)

    joint_amp = ff.joint_spat_temp_conv_amp(
        t_freq, s_freq_x, s_freq_y,
        sf=test_sf, tf=test_tf, collapse_symmetry=False
        )

    assert round(joint_amp, 1) == 62.7  # type: ignore


@given(
    dc=st.one_of([
        st.one_of(st.just(5), st.just(10), st.just(30)),
        st.floats(min_value=0, allow_nan=False, allow_infinity=False)
        ]),
    f1_target=st.floats(min_value=0, allow_nan=False, allow_infinity=False)
    )
def test_estimate_real_amplitude(dc, f1_target):

    dc = 12
    f1_target = 32

    t = Time(np.arange(1000), 'ms')
    opt = est_amp.find_real_f1(dc, f1_target=f1_target)
    estimated_amp = opt.x

    # just reimplementing here ... I know ... but this at least ensures API consistent
    # BUUUT ... as testing an optimisation function, does test that the optimisation
    # is working
    r = est_amp.gen_sin(estimated_amp, dc, t)
    r[r < 0] = 0
    s, _ = est_amp.gen_fft(r, t)
    s = np.abs(s)

    assert np.isclose(s[1], f1_target, atol=1e-3)  # type: ignore


@mark.integration
@mark.parametrize(
    'spat_freq,temp_freq',
    [
        (0, 4),
        (20, 0),  # high spat freq for accurate est at 0 temp_freq
        (1, 1), (2, 2), (4, 8)
    ]
    )
def test_conv_resp_adjustment_process(spat_freq, temp_freq):
    """Test whole convolutional adjustment process is relatively accurate

    Struggles to be accurate for 0 temp frequency.  Because spat_freq must be high
    enough for the stationary grating to integrate the whole RF accurately enough.
    Thus, 20 cpd for spat_freq when temp_freq = 0.
    """

    # get filters from file
    data_dir = do.settings.get_data_dir()
    sf_path = (
        data_dir /
        'Kaplan_et_al_1987_contrast_affects_transmission_fig8A_open_circle-DOGSpatialFilter.pkl')
    tf_path = (
        data_dir /
        'Kaplan_et_al_1987_contrast_affects_transmission_fig_6a_open_circles-TQTempFilter.pkl')

    assert sf_path.exists() and tf_path.exists()

    sf = do.DOGSpatialFilter.load(sf_path)
    tf = do.TQTempFilter.load(tf_path)

    stim_amp = 0.5
    stim_dc = 0.5
    spat_res = ArcLength(1, 'mnt')
    spat_ext = ArcLength(120, 'mnt')
    temp_res = Time(1, 'ms')
    temp_ext = Time(1000, 'ms')
    spat_freq = SpatFrequency(spat_freq)
    temp_freq = TempFrequency(temp_freq)
    orientation = ArcLength(0, 'deg')

    st_params = do.SpaceTimeParams(spat_ext, spat_res, temp_ext, temp_res)
    stim_params = do.GratingStimulusParams(
        spat_freq, temp_freq, orientation=orientation,
        amplitude=stim_amp, DC=stim_dc
    )

    resp = conv.mk_single_sf_tf_response(
        sf=sf, tf=tf, st_params=st_params, stim_params=stim_params,
        rectified=False
        )

    slice_idx = int(0.2 * resp.shape[0])
    resp_slice = resp[slice_idx:]

    resp_rect = resp.copy()
    resp_rect[resp_rect < 0] = 0

    theoretical_resp_params = ff.mk_joint_sf_tf_resp_params(
        stim_params, sf, tf
        )

    spectrum, _ = est_amp.gen_fft(resp_rect, ff.mk_temp_coords(temp_res, temp_ext))
    spectrum = np.abs(spectrum)

    est_DC = resp_slice.min() + ((resp_slice.max() - resp_slice.min()) / 2)

    # if temp_freq is zero, then the steady state will be amp + DC
    expected_amp = (
        theoretical_resp_params.ampitude
        if temp_freq.base > 0
        else
        theoretical_resp_params.ampitude + theoretical_resp_params.DC
        )

    assert np.isclose(
        spectrum[int(temp_freq.hz)], expected_amp,  # type: ignore
        rtol=0.1)

    assert np.isclose(est_DC, theoretical_resp_params.DC, rtol=0.1)  # type: ignore

# > Stimuli


@mark.parametrize(
    'sf,ori,x,y',
    [
        (1, 90.0, 1, 0),
        (1, 30.0, 0.5, -(3**0.5)/2),  # direction of drift (ori - 90)
        (2, 30.0, 1, -(3**0.5))  # test magnitude scales cartesian spat freqs (double here)
    ]
    )
def test_stimuli_cartesion_spat_freq(sf, ori, x, y):
    "Ensure cartesion spatial frequencies correctly derived from direction of drift"

    orientation = ArcLength(ori, 'deg')

    stim = do.GratingStimulusParams(
        spat_freq=SpatFrequency(sf), temp_freq=TempFrequency(1),
        orientation=orientation
        )

    assert (
        np.isclose(stim.spat_freq_x.cpd, x)  # type: ignore
        and
        np.isclose(stim.spat_freq_y.cpd, y)  # type: ignore
        )
