"""Test the creation of filters from empirical data

Probably verging on integration tests also, as much of
filter_functions is relied on in the targetted code base.
"""
from pytest import mark, raises
import hypothesis
from hypothesis import given, strategies as st, assume, event
import numpy as np

from lif.utils.units.units import (
    ArcLength, SpatFrequency, TempFrequency, Time, scalar)
from lif.utils import data_objects as do, exceptions as exc

from lif.receptive_field.filters import (
    filters,
    filter_functions as ff
    )

# > Temporal
def test_fit_tq_temp_filt_success():

    # mock data known to lead to a fit
    fs = TempFrequency(np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]))
    amps = np.array([32, 30, 34, 40, 48, 48, 28, 20, 3])

    data = filters.do.TempFiltData(frequencies=fs, amplitudes=amps)
    opt_res = filters._fit_tq_temp_filt(data)

    assert opt_res.success == True  # noqa: E712


# >> ? test tq_temp_filt provides decent fit
# Not sure of a good way to test if "good" fit ... surely optimisation result
# success is a good enough test?

# > Spatial
def test_fit_dog_spat_filt_success():
    # values known to work
    freq = np.array([
        0.102946, 0.256909, 0.515686, 1.035121, 2.062743, 4.140486, 8.311079,
        16.205212, 33.003893])
    amp = np.array([
        15.648086, 16.727744, 15.764523, 18.014953, 27.488355, 28.952478,
        16.119054, 1.355197, 1.537217])

    data = do.SpatFiltData(
        amplitudes=amp,
        frequencies=SpatFrequency(freq, 'cpd')  # known to be CPD
    )

    opt_res = filters._fit_dog_ft(data)

    assert opt_res.success == True


