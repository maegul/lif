from typing import Optional, Tuple

import brian2
from scipy.ndimage.filters import gaussian_filter1d

from ..receptive_field.filters import filter_functions as ff
from ..receptive_field.filters.filter_functions import (
    do, ArcLength, SpatFrequency, Time, TempFrequency)
from ..receptive_field.filters import estimate_real_amp_from_f1 as est_amp
# from lif.receptive_field.filters.data_objects import DOGSpatialFilter, TQTempFilter
from .. import convolve
from ..utils.data_objects import DOGSpatialFilter, SpaceTimeParams, TQTempFilter

import plotly.express as px
from plotly.subplots import make_subplots

import numpy as np


def dog_sf_ft_hv(
        freqs: SpatFrequency,
        dog_args: do.DOGSpatFiltArgs, spat_res: Optional[ArcLength] = None):
    '''Flips x and y to create orthogonal profiles'''

    if not spat_res:
        spat_res = ArcLength(0.1, 'mnt')

    resp_h = ff.mk_dog_sf_conv_amp(freqs, SpatFrequency(0), dog_args, spat_res)
    resp_v = ff.mk_dog_sf_conv_amp(freqs, SpatFrequency(0), dog_args, spat_res)

    fig = (
        px
        .line(x=freqs.cpd, y=resp_h)
        .update_traces(name='horiz', showlegend=True)
        .add_trace(
            px.line(x=freqs.cpd, y=resp_v)
            .update_traces(
                line_color='red',
                name='vertical',
                showlegend=True
                )
            .data[0]
            )
    )

    return fig


def real_amp_est(r, t, f1_target):
    """Plot results of estimating real amplitude

    Plots signal used for estimate and actual FFT

    Parameters
    ----


    Returns
    ----

    """
    r_rect = r.copy()
    r_rect[r_rect < 0] = 0

    s, f = est_amp.gen_fft(r_rect, t)

    fig = make_subplots(1, 2, column_titles=['Response', 'FFT'])

    fig_r = px.line(x=t.s, y=[r, r_rect])
    _ = fig_r.data[0].update(name='full')
    _ = fig_r.data[1].update(name='rect')

    plot = (
        fig
        .add_traces(fig_r.data, 1, 1)
        .add_trace(px.line(x=f, y=np.abs(s)).data[0], 1, 2)
        .add_annotation(
            x=1, y=f1_target, showarrow=True, arrowhead=1,
            text=f'F1 Amplitude={f1_target}', row=1, col=2,
            ax=50
            )
    )

    return plot


def joint_sf_tf_amp(
        tf: TQTempFilter, sf: DOGSpatialFilter,
        n_increments: int = 20, width=650, height=650):
    '''Plots joint distribution of Response Amplitudes to SF and TF filters

    Joint response from joint_spat_temp_conv_amp()
    '''

    # ===========
    # tf = TQTempFilter.load(TQTempFilter.get_saved_filters()[0])
    # sf = DOGSpatialFilter.load(DOGSpatialFilter.get_saved_filters()[0])
    # -----------
    temp_freqs = tf.source_data.data.frequencies
    spat_freqs_x = sf.source_data.data.frequencies

    temp_freqs = TempFrequency(
        np.linspace(temp_freqs.base.min(), temp_freqs.base.max(), n_increments))
    spat_freqs_x = SpatFrequency(
        np.linspace(spat_freqs_x.base.min(), spat_freqs_x.base.max(), n_increments))

    sf_freq_mg_x, tf_freq_mg = np.meshgrid(spat_freqs_x.base, temp_freqs.hz)  # type: ignore
    sf_freq_mg_x = SpatFrequency(sf_freq_mg_x)
    sf_freq_mg_y = SpatFrequency(np.zeros_like(sf_freq_mg_x.base))
    tf_freq_mg = TempFrequency(tf_freq_mg)

    joint_sf_tf_amp = ff.joint_spat_temp_conv_amp(
        temp_freqs=tf_freq_mg, spat_freqs_x=sf_freq_mg_x, spat_freqs_y=sf_freq_mg_y,
        sf=sf, tf=tf
        )

    fig = make_subplots(
        2, 2,
        shared_xaxes=True, shared_yaxes=True, column_widths=[0.7, 0.3], row_heights=[0.3, 0.7])
    joint_heatmap = (
        px
        .imshow(
            joint_sf_tf_amp, x=spat_freqs_x.cpd, y=temp_freqs.hz, origin='lower',
            labels={'x': 'SF', 'y': 'TF', 'color': 'Amp'}
            )
    )

    spat_freq_source = px.line(
        x=sf.source_data.data.frequencies.cpd, y=sf.source_data.data.amplitudes)

    temp_freq_source = px.line(
        y=tf.source_data.data.frequencies.hz, x=tf.source_data.data.amplitudes)

    base_sf, base_tf = (
        tf.source_data.resp_params.sf.cpd, sf.source_data.resp_params.tf.hz
        )

    plot = (
        fig
        .add_traces(
            [joint_heatmap.data[0], spat_freq_source.data[0], temp_freq_source.data[0]],
            [2, 1, 2], [1, 1, 2])
        .update_xaxes(title='Spat Freq (CPD)', row=2, col=1)
        .update_yaxes(title='Temp Freq (Hz)', row=2, col=1)
        .add_vline(
            x=base_sf, row=1, col=1, annotation_text=f'{base_sf} CPD',
            annotation_position='bottom right')
        .add_hline(
            y=base_tf, row=2, col=2, annotation_text=f'{base_tf} Hz',
            annotation_position='top left')
        .update_layout(width=width, height=height)
    )

    return plot


def poisson_trials_rug(spike_monitor: brian2.SpikeMonitor):
    """Plot rug plot for each spike for each trial

    """

    all_spikes = convolve.aggregate_poisson_trials(spike_monitor)

    plot = (
        px
        .scatter(
            x=all_spikes[1], y=all_spikes[0],
            template='none')
        .update_traces(marker=dict(
            symbol='line-ns', line_width=1, color='black'))
        )

    return plot


def psth(
        st_params: SpaceTimeParams,
        spike_monitor: brian2.SpikeMonitor,
        n_trials: int = 10, bin_width: Time = Time(10, 'ms'),
        sigma: float = 1):

    bins = ff.mk_temp_coords(bin_width, Time(st_params.temp_ext.base + bin_width.base))
    all_spikes = convolve.aggregate_poisson_trials(spike_monitor)

    cnts, cnt_bins = np.histogram(all_spikes[1], bins=bins.base)  # type: ignore

    cnts_hz = cnts / bin_width.s / n_trials
    cnts_smooth = gaussian_filter1d(cnts_hz, sigma=sigma)

    plot = (
        px
        .bar(x=cnt_bins[:-1], y=cnts_hz, template='none')
        .update_traces(marker_color='#bbb')
        .add_trace(
            px.line(x=cnt_bins[:-1], y=cnts_smooth)
            .update_traces(line_color='red')
            .data[0]
            )
        )

    return plot


def mk_sf_orientation_resp(
        spat_freq: SpatFrequency[float], sf_params: do.DOGSpatFiltArgs,
        n_angles: int = 8
        ) -> Tuple[ArcLength[np.ndarray], np.ndarray]:
    """Generates orientation response using spat filt ft methods

    returns orientations used and response array

    orientations are adjusted to represent orientation of stimulus, not direction
    of modulation.  Also wrapped to be within [0, 180].
    """

    angles = ff.mk_even_semi_circle_angles(n_angles)
    # angles represent direction of modulation
    ori_resp = ff.mk_dog_sf_ft(
        *ff.mk_sf_ft_polar_freqs(angles, spat_freq),
        sf_params)

    # adjust angles to represent orientation of stimulus (not modulation direction)
    # and sort so angles and response start at 0 deg
    angles_nda = (angles.deg + 90) % 180
    angles_ord_idx = angles_nda.argsort()

    angles = ArcLength(angles_nda[angles_ord_idx], 'deg')
    ori_resp = ori_resp[angles_ord_idx]

    return angles, ori_resp


def orientation_plot(
        sf_params: do.DOGSpatFiltArgs, spat_freq: SpatFrequency[float],
        n_angles: int = 8):
    """Polar plot of orientation selectivity at spat_freq at n_angles
    evenly spaced orientations

    """

    angles, ori_resp = mk_sf_orientation_resp(spat_freq, sf_params, n_angles)

    # wrap
    resp = np.r_[ori_resp, ori_resp, ori_resp[0]]  # type: ignore
    theta = np.r_[angles.deg, angles.deg + 180, angles.deg[0]]  # type: ignore

    fig = (
            px
            .line_polar(
                r=resp, theta=theta,
                start_angle=0, direction='counterclockwise')
            .update_traces(mode='lines+markers', fill='toself')
        )

    return fig


def ori_spat_freq_heatmap(
        sf_params: do.DOGSpatFiltArgs,
        n_orientations: int = 8,
        n_spat_freqs: int = 50,
        width: int = 500, height: int = 500):
    """Heatmap of with ori on x and spat_freq on y, color ~ response

    """

    angles = ff.mk_even_semi_circle_angles(n_orientations)
    max_spat_freq = ff.find_null_high_sf(sf_params)
    spat_freqs = do.SpatFrequency(np.linspace(0, max_spat_freq.base, n_spat_freqs))

    resps = np.empty((spat_freqs.base.size, angles.base.size))
    for i, ori in enumerate(angles.deg):
        spat_freqs_x, spat_freqs_y = ff.mk_sf_ft_polar_freqs(ArcLength(ori, 'deg'), spat_freqs)
        resps[:, i] = ff.mk_dog_sf_ft(spat_freqs_x, spat_freqs_y, sf_params)

    fig = px.imshow(
            resps, x=angles.deg, y=spat_freqs.cpd,
            labels={'x': 'Orientation (Deg)', 'y': 'Spat Freq (CPD)'},
            origin='lower', width=width, height=height, aspect='auto')
    fig.update_xaxes(tickvals=angles.deg, ticks='outside')  # type: ignore

    return fig


def orientation_circ_var_subplots(
        sf: do.DOGSpatialFilter,
        spat_freq_factors: np.ndarray = np.array([1, 2, 4, 8]),
        circ_vars: np.ndarray = np.arange(0.1, 0.6, 0.1)
        ):
    """Grid of polar subplots of orientation,
    spat_freq along x, circ_var along y, each defining dimensions of spatial filter
    and at what frequency the response has been elicited.

    Spat freq factors determine what spat_freqs used by first estimating the preferred
    spat_freq and multiplying this value by the factors

    Must take Spat Filter and not parameters as need circ variance attributes on filter.
    """

    # Find preferred spatial frequency
    spat_freqs = SpatFrequency(
        np.linspace(
            0,
            ff.find_null_high_sf(sf.parameters).base,
            100)
        )
    sf_resp = ff.mk_dog_sf_ft(
        *ff.mk_sf_ft_polar_freqs(ArcLength(90), spat_freqs),
        sf.parameters)

    max_resp_idx = sf_resp.argmax()
    max_spat_freq = spat_freqs.base[max_resp_idx]
    spat_freqs = SpatFrequency(
        max_spat_freq * spat_freq_factors  # type: ignore
        )

    # Make subplots
    n_cols = spat_freqs.base.size
    n_rows = circ_vars.size

    # prepare figure with appropriate titles
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        start_cell='bottom-left',
        specs=[[{'type': 'polar'}]*n_cols]*n_rows,
        column_titles=[
            (
                f'{round(s, 2)}cpd ({fact}X)'
                if s != max_spat_freq
                else f'<b>{round(s,2)}cpd (pref)</b>'
            )
            for s, fact in zip(spat_freqs.cpd, spat_freq_factors)
            ],
        x_title="Spat Freq", y_title="Circ Var",
        row_titles=[f'{round(cv, 2)}' for cv in circ_vars],
        )
    fig = fig.update_polars(radialaxis_nticks=2, angularaxis_showticklabels=False)  # type: ignore

    # add subplots
    for sfi, spat_f in enumerate(spat_freqs.base):
        for cvi, cv in enumerate(circ_vars):
            ori_fig = orientation_plot(
                sf.mk_ori_biased_parameters(cv),
                do.SpatFrequency(spat_f)
                )
            fig.add_trace(ori_fig.data[0], col=sfi+1, row=cvi+1)

    return fig
