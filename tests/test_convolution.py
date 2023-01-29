"""Convolution and correction being done, tests won't work righ tnow


"""

# from pathlib import Path
# import re
# from typing import cast, Tuple
# from dataclasses import astuple, asdict

# from pytest import mark, raises
# import hypothesis
# from hypothesis import given, strategies as st, assume, event

# from lif.convolution import (
#     convolve as conv,
#     estimate_real_amp_from_f1 as est_amp,
#     correction
#     )
# from lif.utils.units.units import ArcLength, SpatFrequency, TempFrequency, Time, scalar
# from lif.utils import (
#     settings,
#     data_objects as do,
#     exceptions as exc
#     )

# from lif.receptive_field.filters import (
#     filters,
#     filter_functions as ff,
#     cv_von_mises as cvvm
#     )

# import numpy as np
# from numpy import fft  # type: ignore
# from scipy.signal import convolve, argrelmin


# # test joint amp with manual params produces output

# def test_joint_amp():

#     # creating mock parameters from known parameters that work
#     test_sf_params = do.DOGSpatFiltArgs(
#         cent=do.Gauss2DSpatFiltParams(
#             amplitude=36.4265938532914,
#             arguments=do.Gauss2DSpatFiltArgs(
#                 h_sd=ArcLength(value=1.4763319256270793, unit='mnt'),
#                 v_sd=ArcLength(value=1.4763319256270793, unit='mnt'))),
#         surr=do.Gauss2DSpatFiltParams(
#             amplitude=21.123002637615535,
#             arguments=do.Gauss2DSpatFiltArgs(
#                 h_sd=ArcLength(value=6.455530597672735, unit='mnt'),
#                 v_sd=ArcLength(value=6.455530597672735, unit='mnt')
#                 )
#             )
#         )
#     test_tf_params = do.TQTempFiltParams(
#         amplitude=301.92022743003315,
#         arguments=do.TQTempFiltArgs(
#             tau=Time(value=14.90265388080688, unit='ms'),
#             w=11.420030523751624, phi=1.1201854280821189)
#         )

#     test_sf = do.DOGSpatialFilter(
#         source_data=do.SpatFiltParams(
#             data=None,  # type: ignore
#             resp_params=do.SFRespMetaData(dc=15, tf=TempFrequency(value=4, unit='hz'),
#                                           mean_lum=100, contrast=0.5),
#             meta_data=None),
#         parameters=test_sf_params,
#         optimisation_result=None,  # type: ignore
#         ori_bias_params=None)  # type: ignore

#     test_tf = do.TQTempFilter(
#         source_data=do.TempFiltParams(
#             data=None,  # type: ignore
#             resp_params=do.TFRespMetaData(
#                 dc=12, sf=SpatFrequency(value=0.8, unit='cpd'), mean_lum=100, contrast=0.4),
#             meta_data=None
#             ),
#         parameters=test_tf_params,
#         optimisation_result=None)  # type: ignore

#     t_freq = TempFrequency(6.9)
#     s_freq_x = SpatFrequency(3)
#     s_freq_y = SpatFrequency(0)

#     joint_amp = correction.joint_spat_temp_f1_magnitude(
#         t_freq, s_freq_x, s_freq_y,
#         sf=test_sf, tf=test_tf, collapse_symmetry=False
#         )

#     assert round(joint_amp, 1) == 62.7  # type: ignore


# @given(
#     dc=st.one_of([
#         st.one_of(st.just(5), st.just(10), st.just(30)),
#         st.floats(min_value=0, allow_nan=False, allow_infinity=False)
#         ]),
#     f1_target=st.floats(min_value=0, allow_nan=False, allow_infinity=False)
#     )
# def test_estimate_real_amplitude(dc, f1_target):

#     dc = 12
#     f1_target = 32

#     t = Time(np.arange(1000), 'ms')
#     opt = est_amp.find_real_f1(dc, f1_target=f1_target)
#     estimated_amp = opt.x

#     # just reimplementing here ... I know ... but this at least ensures API consistent
#     # BUUUT ... as testing an optimisation function, does test that the optimisation
#     # is working
#     r = est_amp.gen_sin(estimated_amp, dc, t)
#     r[r < 0] = 0
#     s, _ = est_amp.gen_fft(r, t)
#     s = np.abs(s)

#     assert np.isclose(s[1], f1_target, atol=1e-3)  # type: ignore


# @mark.proto
# @mark.integration
# @given(
#     stim_amp = st.floats(min_value=0.1, max_value=1000, allow_nan=False, allow_infinity=False),
#     stim_DC = st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
#     spat_res = st.integers(min_value=30, max_value=2*60), # in seconds of arc
#     temp_res = st.integers(min_value=500, max_value=2_000),  # in microseconds
#     # half CPD should fit within the hardcoded extent?
#     spat_freq = st.floats(min_value=0.5, max_value=10, allow_nan=False, allow_infinity=False),
#     # has to be high enough to fit within the second half of the hard coded extent
#     temp_freq = st.floats(min_value=2, max_value=10, allow_nan=False, allow_infinity=False)
#     )
# def test_estimation_of_temporal_convolution_amplitudes(
#         stim_amp, stim_DC, spat_res, temp_res,
#         spat_freq, temp_freq
#         ):

#     # filters
#     tf = do.TQTempFilter.load(do.TQTempFilter.get_saved_filters()[0])

#     # parameters
#     stim_amp=stim_amp
#     stim_DC=stim_DC
#     spat_res=ArcLength(spat_res, 'sec')
#     temp_res=Time(temp_res, 'us')
#     spat_ext=ArcLength(120, 'mnt')
#     temp_ext=Time(2000, 'ms')

#     orientation = ArcLength(90, 'deg')
#     temp_freq = TempFrequency(temp_freq)
#     spat_freq_x = SpatFrequency(spat_freq)
# # -
# # +
#     # parameter objects
#     st_params = do.SpaceTimeParams(spat_ext, spat_res, temp_ext, temp_res)
#     stim_params = do.GratingStimulusParams(
#         spat_freq_x, temp_freq,
#         orientation=orientation,
#         amplitude=stim_amp, DC=stim_DC
#     )

#     # setup and test for temporal

#     time_coords = ff.mk_temp_coords(st_params.temp_res, st_params.temp_ext)
#     signal = est_amp.gen_sin(
#         amplitude=stim_params.amplitude,
#         DC_amp=stim_params.DC,
#         time=time_coords,
#         freq=stim_params.temp_freq
#         )
#     temp_filter = ff.mk_tq_tf(time_coords, tf.parameters)
#     signal_conv = convolve(signal, temp_filter)[:time_coords.value.size]

#     estimated_amplitude = (
#         stim_params.amplitude *
#         ff.mk_tq_tf_conv_amp(stim_params.temp_freq, tf.parameters, st_params.temp_res)
#         )
#     estimated_DC = (
#         stim_params.DC *
#         ff.mk_tq_tf_conv_amp(TempFrequency(0), tf.parameters, st_params.temp_res)
#         )

#     # remove artefacts from the time constant and "ramping-up" at the beginning of convolution
#     stable_conv = signal_conv[signal_conv.size//2:]

#     # import plotly.express as px
#     # px.line(stable_conv).show()

#     # amplitude is half of total min to maximum
#     actual_amplitude = (stable_conv.max()-stable_conv.min())/2
#     # DC is halfway point between min and max ... or max minus amplitude
#     actual_DC = stable_conv.max() - actual_amplitude

#     assert np.isclose(actual_amplitude, estimated_amplitude, rtol=0.05)
#     assert np.isclose(actual_DC, estimated_DC, rtol=0.05, atol=0.05)


# @mark.proto
# @mark.integration
# @mark.parametrize(
#     'freqs',
#     [
#         (0, 4),
#         (20, 0),  # high spat freq for accurate est at 0 temp_freq
#         (1, 1), (2, 2), (4, 8),
#         # getting into dodgy territory with fractional temp freqs
#         # as the test uses an FFT, which isn't necessarily measuring at these
#         # particular temp_freqs ... so stick with integer temp_freqs
#         # at least test fractional spat freqs and combining high and lows
#         (3.8, 1), (2.3, 2), (1.12, 8)
#     ]
#     )
# # try different amp and DC values too
# @mark.parametrize(
#     'stim_amp,stim_DC',
#     [
#         (1, 1),
#         (0.5, 0.5),
#         (32, -12),
#         (132, 100),
#     ]
#     )
# def test_conv_resp_adjustment_process(
#         freqs,
#         # spat_freq, temp_freq,
#         stim_amp, stim_DC
#         ):
#     """Test whole convolutional adjustment process is relatively accurate

#     Struggles to be accurate for 0 temp frequency.  Because spat_freq must be high
#     enough for the stationary grating to integrate the whole RF accurately enough.
#     Thus, 20 cpd for spat_freq when temp_freq = 0.
#     """

#     # get filters from file
#     data_dir = do.settings.get_data_dir()
#     sf_path = (
#         data_dir /
#         'Kaplan_et_al_1987_contrast_affects_transmission_fig8A_open_circle-DOGSpatialFilter.pkl')
#     tf_path = (
#         data_dir /
#         'Kaplan_et_al_1987_contrast_affects_transmission_fig_6a_open_circles-TQTempFilter.pkl')

#     assert sf_path.exists() and tf_path.exists()

#     sf = do.DOGSpatialFilter.load(sf_path)
#     tf = do.TQTempFilter.load(tf_path)

#     spat_freq, temp_freq = freqs

#     # stim_amp = 1
#     # stim_DC = 1
#     spat_res = ArcLength(1, 'mnt')
#     spat_ext = ArcLength(120, 'mnt')
#     temp_res = Time(1, 'ms')
#     temp_ext = Time(1000, 'ms')
#     spat_freq = SpatFrequency(spat_freq)
#     temp_freq = TempFrequency(temp_freq)
#     orientation = ArcLength(0, 'deg')

#     st_params = do.SpaceTimeParams(spat_ext, spat_res, temp_ext, temp_res)
#     stim_params = do.GratingStimulusParams(
#         spat_freq, temp_freq, orientation=orientation,
#         amplitude=stim_amp, DC=stim_DC
#     )

#     # corrections are made in this function so that the final rectified response
#     # should have the appropriate fourier values for F1 and DC
#     resp = conv.mk_single_sf_tf_response(
#         sf=sf, tf=tf, st_params=st_params, stim_params=stim_params,
#         rectified=False
#         )

#     slice_idx = int(0.2 * resp.shape[0])
#     resp_slice = resp[slice_idx:]

#     resp_rect = resp.copy()
#     resp_rect[resp_rect < 0] = 0

#     theoretical_resp_params = correction.mk_joint_sf_tf_resp_params(
#         stim_params, sf, tf
#         )

#     # use FFT for estimating the F1 amplitude
#     spectrum, _ = est_amp.gen_fft(resp_rect, ff.mk_temp_coords(temp_res, temp_ext))
#     spectrum = np.abs(spectrum)

#     # get DC just from the actual convolution
#     est_DC = resp_slice.min() + ((resp_slice.max() - resp_slice.min()) / 2)

#     # doesn't work as F1 and DC are conflated once rectified
#     # assert np.isclose(est_DC, spectrum[0])

#     # if temp_freq is zero, then the steady state will be amp + DC
#     expected_amp = (
#         theoretical_resp_params.amplitude
#         if temp_freq.base > 0
#         else
#         theoretical_resp_params.amplitude + theoretical_resp_params.DC
#         )
#     expected_dc = theoretical_resp_params.DC

#     assert np.isclose(
#         spectrum[int(temp_freq.hz)], expected_amp,  # type: ignore
#         rtol=0.1)
#     # rounding doesn't work because floating temp freqs just won't get measured
#     # by an FFT as the frequency space is going to be 0, 1, 2 ... with ext 1000ms and res 1ms
#     # assert np.isclose(
#     #     spectrum[int(round(temp_freq.hz,0))], expected_amp,  # type: ignore
#     #     rtol=0.1)

#     assert np.isclose(est_DC, expected_dc, rtol=0.1)  # type: ignore
