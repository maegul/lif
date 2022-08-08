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
from lif.utils import (
    data_objects as do,
    exceptions as exc)

from lif.receptive_field.filters import (
    filters,
    filter_functions as ff,
    cv_von_mises as cvvm
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

# values known to work
mock_sf_freq = np.array([
        0.102946, 0.256909, 0.515686, 1.035121, 2.062743, 4.140486, 8.311079,
        16.205212, 33.003893])
mock_sf_amp = np.array([
        15.648086, 16.727744, 15.764523, 18.014953, 27.488355, 28.952478,
        16.119054, 1.355197, 1.537217])

mock_sf_params = do.SpatFiltParams(
    data=do.SpatFiltData(
        amplitudes=mock_sf_amp,
        frequencies=SpatFrequency(value=mock_sf_freq, unit='cpd')),
    resp_params=do.SFRespMetaData(
        dc=15, tf=TempFrequency(value=4, unit='hz'), mean_lum=100, contrast=0.5),
    meta_data=do.CitationMetaData(
        author='Kaplan_et_al', year=1987,
        title='contrast affects transmission', reference='fig8A_open_circle', doi=None)
    )

mock_sf_args = do.DOGSpatFiltArgs(
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

@mark.proto
def test_fit_dog_spat_filt_success():
    freq = mock_sf_freq
    amp = mock_sf_amp

    data = do.SpatFiltData(
        amplitudes=amp,
        frequencies=SpatFrequency(freq, 'cpd')  # known to be CPD
    )

    opt_res = filters._fit_dog_ft(data)

    assert opt_res.success == True


@mark.proto
def test_circ_var_methods_match():
    """Dumb simple test to ensure that the attributes of two dataclasses match

    yea ... that dumb.

    Idea is that all circ_variance code should be in a separate module for
    circular variance stuff ... but the objects for composition with spatial
    filters should be in the data objects module.
    Here, we test that they contain the same methods for defining the circular
    variance of a receptive field
    """

    assert (
        do.CircularVarianceParams.__dataclass_fields__.keys()
        ==
        cvvm._CircVarSDRatioMethods.__dataclass_fields__.keys()
        )

@mark.proto
def test_ori_biased_lookup_val_creation_success():

    # creating mock parameters from known parameters that work

    cv_params = (
        filters._make_ori_biased_lookup_vals_for_all_methods(mock_sf_args, mock_sf_params)
        )

    assert all(
        cv_method in cv_params.__dir__()
        for cv_method in cvvm.circ_var_sd_ratio_methods._all_methods()
        )

@mark.proto
@mark.integration
def test_spat_filt_creation():


    freq = mock_sf_freq
    amp = mock_sf_amp

    data = do.SpatFiltData(
        amplitudes=amp,
        frequencies=SpatFrequency(freq, 'cpd')  # known to be CPD
    )

    resp_params = do.SFRespMetaData(
        dc=15, tf=TempFrequency(4, 'hz'),
        mean_lum=100, contrast=0.5
        )
    meta_data = do.CitationMetaData(
        author='Kaplan_et_al',
        year=1987,
        title='contrast affects transmission',
        reference='fig8A_open_circle',
        doi=None)
    sf_params = do.SpatFiltParams(
        data = data, resp_params = resp_params, meta_data = meta_data
        )

    sf = filters.make_dog_spat_filt(sf_params)

    # hopefully no errors!
    # assertions??!!

    assert (
        sf.ori_bias_params.ratio2circ_var(4, method='shou') ==
        sf.ori_bias_params.shou.ratio2circ_var(4)
        )




