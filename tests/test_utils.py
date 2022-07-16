from __future__ import annotations
from math import isnan
import inspect
import typing
from typing import Union, Tuple

import numpy as np

from hypothesis import given, assume, strategies as st
from pytest import mark

import lif.utils.units.units as units
from lif.utils.units.units import ArcLength
import lif.utils.data_objects as do
from lif.utils.data_objects import PI

# > units

time_test_units = [
    ('s', 1),
    ('ms', 10**-3),
    ('us', 10**-6)
]


def test_new_unit_methods():
    """methods for changing unit by creating a new instance of the same object
    """

    # for testing new unit factory methods
    @units.dataclass(frozen=True)
    class BasicUnit(units._UnitBC):

        value: float
        unit: str = 'base'
        _base: float = 1
        _other: float = (1/3)  # for good floating point fun

    old = BasicUnit(1, 'base')

    # basic check that unit conversion works fine
    assert old._convert('other') == (BasicUnit._base / BasicUnit._other)

    # new unit works
    new = old.as_new_unit('other')
    assert id(new) != id(old)
    assert new.unit == 'other'
    assert new._convert('base') == old.value
    assert new.value == old._convert('other')

    # in same unit works
    new2 = old.in_same_units_as(new)
    assert id(new2) != id(old)
    assert new2.unit == 'other'
    assert new2._convert('base') == old.value
    assert new2.value == old._convert('other')


@given(
    value=st.one_of(st.integers(), st.floats(allow_infinity=False, allow_nan=False)),
    unit=st.one_of([st.just(val) for val in time_test_units]),
    new_unit=st.one_of([st.just(val) for val in time_test_units])
    )
def test_time_unit(
        value: Union[int, float],
        unit: tuple[str, Union[int, float]], new_unit: tuple[str, Union[int, float]]):

    unit_desc, unit_factor = unit
    new_unit_desc, new_unit_factor = new_unit

    quantity = units.Time(value=value, unit=unit_desc)
    converted_quantity = getattr(quantity, new_unit_desc)

    assert converted_quantity == (value * unit_factor) / new_unit_factor


arclength_test_units = [
    ('deg', 1),
    ('mnt', 1/60),
    ('sec', 1/(60*60)),
    ('rad', 180/PI)
]

# Essentially a replication of the code in Time
# BUT, with separate units used above and hypothesis strategies, should provide
# a safety net

@given(
    value=st.one_of(st.integers(), st.floats(allow_infinity=False, allow_nan=False)),
    unit=st.one_of([st.just(val) for val in arclength_test_units]),
    new_unit=st.one_of([st.just(val) for val in arclength_test_units])
    )
def test_arclength_unit(
        value: Union[int, float],
        unit: tuple[str, Union[int, float]], new_unit: tuple[str, Union[int, float]]):

    unit_desc, unit_factor = unit
    new_unit_desc, new_unit_factor = new_unit

    quantity = units.ArcLength(value=value, unit=unit_desc)
    converted_quantity = getattr(quantity, new_unit_desc)

    assert converted_quantity == (value * unit_factor) / new_unit_factor


@mark.parametrize(
    'value,unit,new_value,new_unit',
    [
        (1, 'deg', 60, 'mnt'),
        (1, 'sec', 1/60, 'mnt'),
        (0.5, 'mnt', 0.5/60, 'deg'),
        (50, 'mnt', (50/60)*PI/180, 'rad'),
        (np.arange(4).reshape(2, 2), 'mnt', np.arange(4).reshape(2, 2)/60, 'deg')
    ]
    )
def test_arclength_unit_simple(value, unit, new_value, new_unit):

    quantity = units.ArcLength(value, unit)

    assert np.allclose(getattr(quantity, new_unit), new_value)  # type: ignore


def test_arclength_multiple():

    multi_vals = [1, 0.3, 100, np.arange(10)]
    unit = 'sec'

    multi_arclength = units.ArcLength.mk_multiple(multi_vals, unit=unit)
    simple_multi_arclength = tuple(
        units.ArcLength(val, unit=unit)
        for val in multi_vals
        )

    assert multi_arclength == simple_multi_arclength


spat_freq_test_units = [
    ('cpd', 1),
    ('cpm', 60),
    ('cpd_w', 1/(2*PI))
]


@given(
    value=st.one_of(st.integers(), st.floats(allow_infinity=False, allow_nan=False)),
    unit=st.one_of([st.just(val) for val in spat_freq_test_units]),
    new_unit=st.one_of([st.just(val) for val in spat_freq_test_units])
    )
def test_spat_frequency_unit(
        value: Union[int, float], unit: tuple[str, Union[int, float]],
        new_unit: tuple[str, Union[int, float]]):

    unit_desc, unit_factor = unit
    new_unit_desc, new_unit_factor = new_unit

    quantity = units.SpatFrequency(value=value, unit=unit_desc)
    converted_quantity = getattr(quantity, new_unit_desc)

    assert converted_quantity == (value * unit_factor) / new_unit_factor


temp_freq_test_units = [
    ('hz', 1),
    ('w', 1/(2*PI))
]


@given(
    value=st.one_of(st.integers(), st.floats(allow_infinity=False, allow_nan=False)),
    unit=st.one_of([st.just(val) for val in temp_freq_test_units]),
    new_unit=st.one_of([st.just(val) for val in temp_freq_test_units])
    )
def test_temp_frequency_unit_simple(
        value: Union[int, float], unit: tuple[str, Union[int, float]],
        new_unit: tuple[str, Union[int, float]]):

    unit_desc, unit_factor = unit
    new_unit_desc, new_unit_factor = new_unit

    quantity = units.TempFrequency(value=value, unit=unit_desc)
    converted_quantity = getattr(quantity, new_unit_desc)

    assert converted_quantity == (value * unit_factor) / new_unit_factor



# @mark.parametrize("unit", list(units.Time.factors.keys()))  # test for each factor
# @given(
#     value=st.one_of(  # test both integers and floats
#         st.integers(),
#         st.floats()
#     ))
# def test_time_unit_conversion(value: Union[float, int], unit: str):
#     """test conversion between units works"""
#     assume(not isnan(value))  # don't care about nans ... assume good

#     quantity = units.Time(value, unit=unit)
    
#     for factor in units.Time.factors:
#         # access property
#         new_value = getattr(quantity, factor)

#         # obsessive typing here ... casting new_value to be that of the property return type
#         factor_prop_sig = inspect.signature(getattr(quantity.__class__, factor).fget)
#         factor_return_type = factor_prop_sig.return_annotation
#         typing.cast(factor_return_type, new_value)

#         # original value, 
#         # scaled to base unit (* factors[unit]) then scaled to target unit (/ factors[factor])
#         assert_value = (value * units.Time.factors[unit]) / units.Time.factors[factor]

#         assert new_value == assert_value

# > Data Objects

# >> DOG Spat Filt

# >>> Data object creation veracity

def test_gauss_params_round_trip():

    gauss_params = do.Gauss2DSpatFiltParams(
        amplitude=5,
        arguments=do.Gauss2DSpatFiltArgs(
            h_sd=ArcLength(3), v_sd=ArcLength(11)
            )
        )

    assert gauss_params == do.Gauss2DSpatFiltParams.from_iter(gauss_params.array())  # type: ignore


def test_dog_spat_filt_1d_round_trip():

    params = do.DOGSpatFiltArgs1D(
        cent=do.Gauss1DSpatFiltParams(1, ArcLength(11)),
        surr=do.Gauss1DSpatFiltParams(0.5, ArcLength(22))
        )

    assert params == do.DOGSpatFiltArgs1D.from_iter(params.array())


# >>> Orientation Biased Creation methods

@mark.parametrize('arclength_unit', ('mnt', 'deg', 'sec'))
def test_gauss2d_spat_filt_args_ori_biased_duplicate(arclength_unit):
    h_sd, v_sd = (ArcLength(v, arclength_unit) for v in (20, 20))
    gauss_2d_args = do.Gauss2DSpatFiltArgs(h_sd=h_sd, v_sd=v_sd)

    v_sd_factor, h_sd_factor = 2, 0.5
    ori_biased_gauss_args = gauss_2d_args.mk_ori_biased_duplicate(
        v_sd_factor=v_sd_factor, h_sd_factor=h_sd_factor)

    # different objects (prob does test for deepcopy success)
    assert gauss_2d_args != ori_biased_gauss_args

    # values are correct
    assert (
        ori_biased_gauss_args.v_sd.base
        ==
        gauss_2d_args.v_sd.base * v_sd_factor
        )
    assert (
        ori_biased_gauss_args.h_sd.base
        ==
        gauss_2d_args.h_sd.base * h_sd_factor
        )

    # units are the same
    assert (
        ori_biased_gauss_args.v_sd.unit
        ==
        gauss_2d_args.v_sd.unit
        )
    assert (
        ori_biased_gauss_args.h_sd.unit
        ==
        gauss_2d_args.h_sd.unit
        )


@mark.parametrize('arclength_unit', ('mnt', 'deg', 'sec'))
def test_spat_filt_args_ori_biased_duplicate(arclength_unit):
    mag_cent, cent_h_sd, cent_v_sd = 1, 20, 20
    mag_surr, surr_h_sd, surr_v_sd = 1, 100, 100

    dog_rf_params = do.DOGSpatFiltArgs(
        cent=do.Gauss2DSpatFiltParams.from_iter(
            [mag_cent, cent_h_sd, cent_v_sd],
            arclength_unit=arclength_unit),
        surr=do.Gauss2DSpatFiltParams.from_iter(
            [mag_surr, surr_h_sd, surr_v_sd],
            arclength_unit=arclength_unit)
        )

    v_sd_factor, h_sd_factor = 2, 0.5
    ori_biased_dog_rf = dog_rf_params.mk_ori_biased_duplicate(
        v_sd_factor=v_sd_factor, h_sd_factor=h_sd_factor)

    # different objects (prob does NOT test for deepcopy success)
    assert ori_biased_dog_rf != dog_rf_params

    # values are correct
    assert (
        ori_biased_dog_rf.cent.arguments.v_sd.base
        ==
        dog_rf_params.cent.arguments.v_sd.base * v_sd_factor
        )
    assert (
        ori_biased_dog_rf.cent.arguments.h_sd.base
        ==
        dog_rf_params.cent.arguments.h_sd.base * h_sd_factor
        )

    # units are the same
    assert (
        ori_biased_dog_rf.cent.arguments.v_sd.unit
        ==
        dog_rf_params.cent.arguments.v_sd.unit
        )
    assert (
        ori_biased_dog_rf.cent.arguments.h_sd.unit
        ==
        dog_rf_params.cent.arguments.h_sd.unit
        )
