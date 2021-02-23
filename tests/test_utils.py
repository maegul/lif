from math import isnan

from hypothesis import given, assume, strategies as st
from pytest import mark

import lif.utils.units as units


# > units

@mark.parametrize("unit", list(units.Time.factors.keys()))  # test for each factor
@given(
    value=st.one_of(  # test both integers and floats 
        st.integers(),
        st.floats()
    ))
def test_time_unit_conversion(value, unit):
    """test conversion between units works"""
    assume(not isnan(value))  # don't care about nans ... assume good

    quantity = units.Time(value, unit=unit)  
    
    for factor in units.Time.factors:
        # access property
        new_value = getattr(quantity, factor)
        # original value, 
        # scaled to base unit (* factors[unit]) then scaled to target unit (/ factors[factor])
        assert_value = (value * units.Time.factors[unit]) / units.Time.factors[factor]

        assert new_value == assert_value
