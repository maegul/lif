from math import isnan

from hypothesis import given, assume, strategies as st
from pytest import mark

import lif.utils.units as units


# > units
@mark.parametrize("unit", list(units.Time.factors.keys()))
@given(
    value=st.one_of(
        st.integers(),
        st.floats()
    ))
def test_time_unit_conversion(value, unit):
    """test conversion between units works"""
    assume(not isnan(value))

    quantity = units.Time(value, unit=unit)

    for factor in units.Time.factors:
        new_value = getattr(quantity, factor)
        assert new_value == (value * units.Time.factors[unit]) / units.Time.factors[factor]
