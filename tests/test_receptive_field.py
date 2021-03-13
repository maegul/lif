from lif.utils.units.units import TempFrequency
from pytest import mark
from hypothesis import given, strategies as st

from lif.receptive_field.filters import filters, filter_functions as ff, data_objects as do

import numpy as np
from numpy import fft


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


# > Temp Filters


t = filters.do.TQTempFiltParams(
    amplitude=44, 
    arguments=filters.do.TQTempFiltArgs(
        tau=15, w=3, phi=0.3))


def test_tq_params_dict_conversion():
    putative_dict = {
        'amplitude': t.amplitude,
        'tau': t.arguments.tau,
        'w': t.arguments.w,
        'phi': t.arguments.phi
    }

    assert putative_dict == t.to_flat_dict()


def test_tq_params_array_round_trip():

    assert t == do.TQTempFiltParams.from_iter(t.array())


def test_fit_tq_temp_filt():

    # mock data known to lead to a fit
    fs = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64])
    amps = np.array([32, 30, 34, 40, 48, 48, 28, 20, 3])

    data = filters.do.TempFiltData(frequencies=fs, amplitudes=amps)
    opt_res = filters._fit_tq_temp_filt(data)
    
    assert opt_res.success == True  # noqa: E712


basic_float_strat = st.floats(min_value=1, max_value=10, allow_infinity=False, allow_nan=False)


# > Spatial Filters

def test_gauss_params_round_trip():

    gauss_params = do.Gauss2DSpatFiltParams(
        amplitude=5,
        arguments=do.Gauss2DSpatFiltArgs(
            h_sd=3, v_sd=11
            )
        )

    assert gauss_params == do.Gauss2DSpatFiltParams.from_iter(gauss_params.array())


def test_dog_spat_filt_1d_round_trip():

    params = do.DOGSpatFiltArgs1D(
        cent=do.Gauss1DSpatFiltParams(1, 11),
        surr=do.Gauss1DSpatFiltParams(0.5, 22)
        )

    assert params == do.DOGSpatFiltArgs1D.from_iter(params.array())

@given(
    x_sd=basic_float_strat,
    y_sd=basic_float_strat,
    mag=basic_float_strat
    )
def test_gauss_2d_sum(x_sd, y_sd, mag):
    # spat_res = 1
    # x_sd, y_sd = 10, 20
    # mag = 1

    gauss_params = do.Gauss2DSpatFiltParams(
        amplitude=mag,
        arguments=do.Gauss2DSpatFiltArgs(x_sd, y_sd))

    # ensure resolution high enough for sd
    spat_res = (np.floor(np.min([x_sd, y_sd]))) / 10
    # ensure extent high enough for capture full gaussian
    spat_ext = (2 * 5*np.ceil(np.max([x_sd, y_sd])) + 1)  # type: ignore

    xc, yc = ff.mk_coords(temp_dim=False, spat_res=spat_res, spat_ext=spat_ext)

    gauss_2d = ff.mk_gauss_2d(xc, yc, gauss_params=gauss_params)

    assert np.isclose(gauss_2d.sum()*spat_res**2, mag)


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

    # preparing params and coords
    dog_rf_params = do.DOGSpatFiltArgs(
        cent=do.Gauss2DSpatFiltParams.from_iter([mag_cent, cent_h_sd, cent_v_sd]),
        surr=do.Gauss2DSpatFiltParams.from_iter([mag_surr, surr_h_sd, surr_v_sd])
        )

    # ensure resolution high enough for sd
    spat_res = (np.floor(np.min([cent_h_sd, cent_v_sd, surr_h_sd, surr_v_sd]))) / 10  # type: ignore
    # ensure extent high enough for capture full gaussian
    spat_ext = (
        2 * 5*np.ceil(np.max([cent_h_sd, cent_v_sd, surr_h_sd, surr_v_sd])) + 1)  # type: ignore

    xc, yc = ff.mk_coords(temp_dim=False, spat_res=spat_res, spat_ext=spat_ext)

    # making dog rf
    dog_rf = ff.mk_dog_rf(xc, yc, dog_args=dog_rf_params)

    # making rf with direct code

    rf_cent = (
        # mag divide by normalising factor with both sds (equivalent to sq if they were identical)
        (mag_cent / (2 * np.pi * cent_v_sd * cent_h_sd)) *
        np.exp(
            - (
                (xc**2 / (2 * cent_h_sd**2)) +
                (yc**2 / (2 * cent_v_sd**2))
            )
        )
    )
    rf_surr = (
        (mag_surr / (2 * np.pi * surr_v_sd * surr_h_sd)) *
        np.exp(
            - (
                (xc**2 / (2 * surr_h_sd**2)) +
                (yc**2 / (2 * surr_v_sd**2))
            )
        )
    )

    rf = rf_cent - rf_surr

    assert np.allclose(rf, dog_rf)


# Do I need to use hypothesis for this?  Main point is that any equivalence is good?
# @mark.parametrize(
#         'cent_h_sd,cent_v_sd,surr_h_sd,surr_v_sd,mag_cent,mag_surr',
#         [(10, 13, 30, 30, 17/16, 15/16)]
#     )
cent_sd_strat = st.floats(min_value=10, max_value=29, allow_infinity=False, allow_nan=False)
surr_sd_strat = st.floats(min_value=30, max_value=50, allow_infinity=False, allow_nan=False)
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

    dog_rf_params = do.DOGSpatFiltArgs(
        cent=do.Gauss2DSpatFiltParams.from_iter([mag_cent, cent_h_sd, cent_v_sd]),
        surr=do.Gauss2DSpatFiltParams.from_iter([mag_surr, surr_h_sd, surr_v_sd])
        )

    # ensure resolution high enough for sd
    spat_res = (np.floor(np.min([cent_h_sd, cent_v_sd, surr_h_sd, surr_v_sd]))) / 10  # type: ignore
    # ensure extent high enough for capture full gaussian
    spat_ext = (
        2 * 5*np.ceil(np.max([cent_h_sd, cent_v_sd, surr_h_sd, surr_v_sd])) + 1)  # type: ignore

    xc, yc = ff.mk_coords(temp_dim=False, spat_res=spat_res, spat_ext=spat_ext)

    dog_rf = ff.mk_dog_rf(xc, yc, dog_args=dog_rf_params)

    dog_rf_fft = np.abs(fft.fftshift(fft.fft2(dog_rf)))
    dog_rf_fft_freqs = fft.fftshift(fft.fftfreq(xc.shape[0], d=spat_res))

    fx, fy = (TempFrequency(f) for f in np.meshgrid(dog_rf_fft_freqs, dog_rf_fft_freqs))
    # FT = T * FFT ~ amplitude of convolution
    dog_rf_ft = ff.mk_dog_rf_ft(fx, fy, dog_args=dog_rf_params) / spat_res

    # atol adjusted as there's some systemic error in comparing continuous with FFT DFT
    # check graphs of the difference to see

    # hopefully this isn't too much of a problem!

    is_close = np.isclose(dog_rf_ft, dog_rf_fft, atol=1e-5)
    assert is_close.mean() > 0.95  # type: ignore


@given(
    mag_cent=basic_float_strat,
    mag_surr=basic_float_strat,
    cent_h_sd=basic_float_strat,
    # cent_v_sd=basic_float_strat,
    surr_h_sd=basic_float_strat,
    # surr_v_sd=basic_float_strat
    )
def test_dog_ft_2d_to_1d(
        cent_h_sd: float,
        surr_h_sd: float,
        mag_cent: float, mag_surr: float):
    """Compare simple 1d continuous fourier with slice of 2d

    If this passes, then direct mapping from 2d to 1d fourer transform for gaussians

    1d params:cent_amp, surr_amp, cent_sd, surr_sd
    -> 2d params: h_sd, v_sd = sd (as radially symmetrical)

    Works because if all y frequencies (fy) are zero (because of the slice) then this term
    in the equation has no impact
    """

    dog_sf_args = do.DOGSpatFiltArgs(
        cent=do.Gauss2DSpatFiltParams(
            amplitude=mag_cent,
            arguments=do.Gauss2DSpatFiltArgs(h_sd=cent_h_sd, v_sd=cent_h_sd)
            ),
        surr=do.Gauss2DSpatFiltParams(
            amplitude=mag_surr,
            arguments=do.Gauss2DSpatFiltArgs(h_sd=surr_h_sd, v_sd=surr_h_sd)
            )
        )

    # ensure resolution high enough for sd
    spat_res = (np.floor(np.min([cent_h_sd, surr_h_sd]))) / 10  # type: ignore
    # ensure extent high enough for capture full gaussian
    spat_ext = (
        2 * 5*np.ceil(np.max([cent_h_sd, surr_h_sd])) + 1)  # type: ignore

    # spat_res, spat_ext = 1, 300
    coords = ff.mk_spat_coords_1d(spat_res=spat_res, spat_ext=spat_ext)
    ft_freq = fft.fftshift(fft.fftfreq(coords.size, d=spat_res))
    fx, fy = (TempFrequency(f) for f in np.meshgrid(ft_freq, ft_freq))  # type: ignore

    # 2d ft
    dog_ft = ff.mk_dog_rf_ft(fx, fy, dog_args=dog_sf_args)
    # take center slice
    x_cent_dog_ft = dog_ft[int(coords.size//2), :]

    # make 1d for cent and surr
    cent_ft = ff.mk_gauss_1d_ft(
        TempFrequency(ft_freq),
        amplitude=dog_sf_args.cent.amplitude, sd=dog_sf_args.cent.arguments.h_sd)
    surr_ft = ff.mk_gauss_1d_ft(
        TempFrequency(ft_freq),
        amplitude=dog_sf_args.surr.amplitude, sd=dog_sf_args.surr.arguments.h_sd)

    # make 1d dog ft (by subtracting)
    dog_1d_ft = cent_ft - surr_ft

    assert np.allclose(x_cent_dog_ft, dog_1d_ft)  # type: ignore
