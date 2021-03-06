from __future__ import annotations
from lif.receptive_field.filters.data_objects import PI
from math import isnan
import inspect
import typing
from typing import Union, Tuple

import numpy as np

from hypothesis import given, assume, strategies as st
from pytest import mark

import lif.utils.units.units as units

# > units

time_test_units = [
    ('s', 1),
    ('ms', 10**-3),
    ('us', 10**-6)
]

# Essentially a replication of the code in Time
# BUT, with separate units used above and hypothesis strategies, should provide
# a safety net


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
