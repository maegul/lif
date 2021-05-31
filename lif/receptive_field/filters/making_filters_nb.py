# > Imports
# ===========
from typing import cast
# -----------

# ===========
from lif.utils.units.units import ArcLength, SpatFrequency, TempFrequency, Time
from lif.receptive_field.filters import (
    data_objects as do, filter_functions as ff, filters)
# -----------


# > Temp Filter
# >> clean up temp data from tracey (Kaplan 1987)
# ===========
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as psp
# -----------

# >> Data directory
# ===========
import lif.utils.settings as settings
data_dir = settings.get_data_dir()
# -----------

# >> Processing raw temp_filt data from tracey
# ===========
tf_raw_path = data_dir / 'Kaplan_et_al_1987_fig6A.csv'
tf_raw_data = pd.read_csv(tf_raw_path, header=None)
# -----------
# ===========
clean_tf_raw_data: pd.DataFrame = 2**tf_raw_data.T
clean_tf_raw_data.rename(columns={0: 'tf_hz', 1: 'resp_imp_s'}, inplace=True)
clean_tf_raw_data
# -----------
# ===========
clean_tf_raw_data.to_csv(data_dir/'Kaplan_et_al_1987_fig6A_clean.csv', index=False)
# -----------

# >> Loading cleaned data from file in data dir
# ===========
clean_tf_raw_data = pd.read_csv(data_dir/'Kaplan_et_al_1987_fig6A_clean.csv')
# -----------

# >> Preparing data for tq tf fit
# ===========
amps = clean_tf_raw_data.resp_imp_s.values
fs = do.TempFrequency(clean_tf_raw_data.tf_hz.values, 'hz')
# -----------
# ===========
data = do.TempFiltData(fs, amps)
resp_params = do.TFRespMetaData(
    dc=12, contrast=0.4, mean_lum=100,
    sf=do.SpatFrequency(0.8, 'cpd')
    )
meta_data = do.CitationMetaData(
    'Kaplan et al', 1987,
    'contrast affects transmission', 'fig 6a open circles')
# -----------
# ===========
tf_params = do.TempFiltParams(
    data=data,
    resp_params=resp_params,
    meta_data=meta_data
    )
# -----------

# >> Fitting tq tf to raw data
# ===========
tqtf = filters.make_tq_temp_filt(parameters=tf_params)
# -----------
# ===========
tqtf.parameters
# -----------

# >> Graphing results
# ===========
time = ff.mk_temp_coords(temp_res=ff.Time(1, 'ms'), temp_ext=ff.Time(200, 'ms'))
tf = ff.mk_tq_tf(t=time, tf_params=tqtf.parameters)
# -----------
# ===========
freqs = tqtf.source_data.data.frequencies
tf_ft = ff.mk_tq_tf_ft(freqs=freqs, tf_params=tqtf.parameters)
# -----------
# ===========

filt = px.line(x=time.ms, y=tf, labels={'x': 'Time (ms)', 'y': 'Mag'})

ft = px.line(x=freqs.hz, y=[tf_ft, tqtf.source_data.data.amplitudes])
ft.data[0].name = 'tq'
ft.data[1].name = 'source'

sp = psp.make_subplots(1, 2, subplot_titles=['filter', 'fourier/data'])
sp.add_trace(filt.data[0], 1, 1).add_traces(ft.data, 1, 2).show()
# -----------

# >> Testing for single frequency
# ===========
ff.mk_tq_tf_ft(freqs=ff.TempFrequency(10), tf_params=tqtf.parameters)
# -----------

# >> save Filter
# ===========
tqtf.save()
# -----------

# >> load filter
# ===========
loaded_tqtf = do.TQTempFilter.load(do.TQTempFilter.get_saved_filters()[0])
# -----------
# ===========
loaded_tqtf.parameters
# -----------
# ===========
tqtf = loaded_tqtf
# -----------

# > Spat Filt

# >> Clean CSV Data
# ===========
sf_data_raw = data_dir / 'Kaplan_et_al_1987_fig8A.csv'
# -----------
# ===========
sf_data_raw_df = pd.read_csv(sf_data_raw)
# -----------
# ===========
px.line(sf_data_raw_df, x='freq', y='amp').update_traces(mode='lines+markers').show()
# -----------

# >> FItting
# ===========
data = do.SpatFiltData(
    amplitudes=sf_data_raw_df.amp.values,
    frequencies=SpatFrequency(sf_data_raw_df.freq.values, 'cpd')
    )
# -----------
# ===========
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
# -----------
# ===========
sf_params = do.SpatFiltParams(
    data = data, resp_params = resp_params, meta_data = meta_data
    )
# -----------
# ===========
sf = filters.make_dog_spat_filt(sf_params)
# -----------
# ===========
sf.parameters
# -----------


# >> Testing and Graphing

# >>> Fourier
# ===========
stim_theta = ArcLength(0)
freqs = SpatFrequency(np.linspace(0,20,100))

freqs_x, freqs_y = ff.mk_sf_ft_polar_freqs(stim_theta, freqs)
sf_ft = ff.mk_dog_sf_ft(freqs_x, freqs_y, sf.parameters)
# -----------
# ===========
(
    px
    .line(sf_data_raw_df, x='freq', y='amp')
    .update_traces(mode='lines+markers')
    .add_trace(
        px
        .line(x=freqs.cpd, y=sf_ft)
        .update_traces(name='model', line_color='red')
        .data[0]
        )
).show()
# -----------

# >>> Receptive Field
# ===========
sc_x, sc_y = ff.mk_spat_coords(spat_ext=ArcLength(50, 'mnt'), spat_res=ArcLength(0.5, 'mnt'))
rf = ff.mk_dog_sf(sc_x, sc_y, sf.parameters)
# -----------
# ===========
px.imshow(
    rf,
    color_continuous_scale=px.colors.diverging.RdBu_r,
    color_continuous_midpoint=0
    ).show()
# -----------
# ===========
cent = ff.mk_gauss_2d(sc_x, sc_y, sf.parameters.cent)
surr = ff.mk_gauss_2d(sc_x, sc_y, sf.parameters.surr)
# -----------
# ===========
px.imshow(cent, title='Cent').show()
px.imshow(surr, title='Surround').show()
# -----------

# >> Saving
# ===========
sf.save(overwrite=True)
# -----------

# >> Loading
# ===========
saved_sfs = do.DOGSpatialFilter.get_saved_filters()
# saved_sfs
# -----------
# ===========
sf = do.DOGSpatialFilter.load(saved_sfs[0])
# -----------
# ===========
sf
# -----------




# save this proto code so that reproducible
# make final changes to filter data objects
# clean up data dir

# Prepare final LGN model
    # have both spatial and temporal filter
    # combine both for single magnitude
    # prepare magnitudes from stim params (don't need stim for this)




# # ===========
# fs = TempFrequency(np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]))
# amps = np.array([32, 30, 34, 40, 48, 48, 28, 20, 3])

# data = filters.do.TempFiltData(frequencies=fs, amplitudes=amps)
# opt_res = filters._fit_tq_temp_filt(data)
# # -----------
# # ===========

# # -----------




# # ===========
# sd = 10
# spat_res = 2
# coords = mk_sd_limited_spat_coords(sd, spat_res = spat_res)
# t_gauss = mk_gauss_1d(coords, sd=sd)
# # -----------
# # ===========
# plt.plot(coords, t_gauss)
# # -----------
# # ===========
# t_gauss.sum()
# # -----------
# # ===========
# tg_ft = fft.rfft(t_gauss)
# tg_ft_freq = fft.rfftfreq(coords.size, d=spat_res)
# # -----------
# # ===========
# plt.clf()
# # -----------
# # ===========
# plt.plot(tg_ft_freq, np.abs(tg_ft))
# # -----------
# # ===========
# freq = Frequency(tg_ft_freq)
# # -----------
# # ===========
# c_ft = mk_gauss_1d_ft(freqs=freq, spat_res=spat_res, sd=sd)
# # -----------
# # ===========
# plt.plot(freq.hz, c_ft / spat_res)
# # -----------
# # ===========
# plt.legend(labels=['np.fft', 'custom ft'])
# # -----------


# # ===========
# spat_res = 1
# x_sd, y_sd = 10, 20
# spat_ext = (2 * 5*np.ceil(np.max([x_sd, y_sd])) + 1)
# xc, yc = mk_coords(temp_dim=False, spat_res=spat_res, spat_ext=spat_ext)
# # -----------
# # ===========
# gauss_x = mk_gauss_1d(xc, sd=x_sd)
# gauss_y = mk_gauss_1d(yc, sd=y_sd)
# # -----------
# # ===========
# gauss_2d = gauss_x * gauss_y
# # -----------
# # ===========
# gauss_2d.shape
# # -----------
# # ===========
# plt.clf()
# plt.imshow(gauss_2d)
# plt.colorbar()
# # -----------
# # ===========
# # compare above with mk_rf()
# rf, rf_cent, rf_surr = mk_rf(
#     spat_res = spat_res,
#     cent_h_sd = 10, cent_v_sd = 20,
#     surr_h_sd = 10, surr_v_sd = 20,
#     return_cent_surr = True
#     )
# # -----------
# # ===========
# rf.shape
# # -----------
# # ===========
# plt.clf()
# plt.imshow(rf_cent)
# plt.colorbar()
# # -----------
# # ===========
# plt.clf()
# plt.imshow(rf_cent-gauss_2d)
# plt.colorbar()
# # -----------
# # ===========
# np.mean(rf_cent == gauss_2d)
# # -----------
# # ===========
# np.allclose(gauss_2d, rf_cent)
# # -----------

# # ===========
# test_gauss_args = do.Gauss2DSpatFiltArgs(1, 10, 20)
# # -----------
# # ===========
# g2d = mk_gauss_2d(xc, yc, gauss_args=test_gauss_args)
# # -----------
# # ===========
# g2d.sum()
# # -----------
# # ===========
# plt.clf()
# plt.imshow(g2d)
# plt.colorbar()
# # -----------
# # ===========
# g2d_fft = fft.fft2(g2d)
# # -----------
# # ===========
# plt.clf()
# plt.imshow(np.abs(fft.fftshift(g2d_fft)))
# plt.colorbar()
# # -----------
# # ===========
# g2d_fft_freq = fft.rfftfreq(xc.shape[0], d=spat_res)
# # -----------
# # ===========
# g2d_fft_freq = fft.fftshift(fft.fftfreq(xc.shape[0], d=spat_res))
# # -----------
# # ===========
# fx, fy = np.meshgrid(g2d_fft_freq, g2d_fft_freq)
# # -----------
# # ===========
# g2d_ft = mk_gauss_2d_ft(freqs_x=Frequency(fx), freqs_y=Frequency(fy), spat_res=spat_res, gauss_args=test_gauss_args)
# # -----------
# # ===========
# g2d_ft.shape
# # -----------
# # ===========
# plt.clf()
# plt.imshow(g2d_ft)
# plt.colorbar()
# # -----------
# # ===========
# np.allclose(g2d_ft, np.abs(fft.fftshift(g2d_fft)), atol=1e-5)
# # -----------
# # ===========
# ft_diff = g2d_ft - np.abs(fft.fftshift(g2d_fft))
# # -----------
# # ===========
# plt.clf()
# plt.imshow(ft_diff)
# plt.colorbar()
# # -----------



# # spat filt fitting

# # ===========
# from lif.receptive_field.filters import filters, filter_functions as ff, data_objects as do
# # -----------
# # ===========
# import pandas as pd
# import matplotlib.pyplot as plt
# # -----------
# # ===========
# d = pd.read_csv('/Users/errollloyd/.lif_hws_data/Kaplan_et_al_1987_fig8A.csv')
# # -----------
# # ===========
# sfdata = do.SpatFiltData(amplitudes=d.amp.values, frequencies=d.freq.values)
# respmetadata = do.SFRespMetaData(dc=15, tf=4, mean_lum=100)
# metadata = do.CitationMetaData(
#     author='Kaplan_et_al', title='contrast affects transmission', year=1987, reference='fig8A_open_circle')
# sfparams = do.SpatFiltParams(data=sfdata, resp_params=respmetadata, meta_data=metadata)
# # -----------
# # ===========
# sf = filters.make_dog_spat_filt(sfparams)
# # -----------
# # ===========
# sf.save()
# # -----------
# # ===========
# dog_ft_1d = ff.mk_dog_rf_ft_1d(freqs=do.SpatFrequency(d.freq.values), dog_args = sf.parameters)
# # -----------
# # ===========
# plt.plot(d.freq.values, ff.mk_dog_rf_ft_1d(freqs=do.SpatFrequency(d.freq.values), dog_args = sf.parameters))
# # -----------
# # ===========
# plt.plot(d.freq, d.amp)
# # -----------


# # temp filter

# # ===========
# import numpy as np
# # -----------
# # ===========
# fs = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64])
# amps = np.array([32, 30, 34, 40, 48, 48, 28, 20, 3])
# tfdata = do.TempFiltData(frequencies=fs, amplitudes=amps)
# # -----------
# # ===========
# tf_resp_metadata = do.TFRespMetaData(dc=15, sf=0.8, mean_lum=100, contrast=0.40)
# # -----------
# # ===========
# cite_md = do.CitationMetaData(author='Kaplan_et_al', year=1987, title='contrast affects transmission', reference='fig6a_open_circle')
# # -----------
# # ===========
# tfparams = do.TempFiltParams(data=tfdata, resp_params=tf_resp_metadata, meta_data=cite_md)
# # -----------
# # ===========
# tf = filters.make_tq_temp_filt(tfparams)
# # -----------
# # ===========
# tf.parameters
# # -----------
# # ===========
# tqtf_ft = ff.mk_tq_ft(fs*2*ff.PI,
#     tau=tf.parameters.arguments.tau, w=tf.parameters.arguments.w, phi=tf.parameters.arguments.phi)
# # -----------
# # ===========
# plt.figure()
# plt.plot(fs, tf.parameters.amplitude* tqtf_ft)
# # -----------
# # ===========
# plt.plot(fs, amps)
# # -----------
# # ===========

# # -----------