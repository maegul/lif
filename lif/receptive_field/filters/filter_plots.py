from lif.receptive_field.filters.data_objects import DOGSpatialFilter, TQTempFilter
from typing import Optional

from . import filter_functions as ff
from .filter_functions import do, ArcLength, SpatFrequency, Time, TempFrequency
from . import estimate_real_amp_from_f1 as est_amp

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

    (
        fig
        .add_traces(fig_r.data, 1, 1)
        .add_trace(px.line(x=f, y=np.abs(s)).data[0], 1, 2)
        .add_annotation(
            x=1, y=f1_target, showarrow=True, arrowhead=1,
            text=f'F1 Amplitude={f1_target}', row=1, col=2,
            ax=50
            )
        .show()
    )


def joint_sf_tf_amp(
    tf: TQTempFilter, sf: DOGSpatialFilter,
    n_increments: int = 20, width=650, height=650):
    '''Plots joint distribution of Response Amplitudes to SF and TF filters

    Joint response from joint_spat_temp_conv_amp()
    '''

    # # ===========
    # tf = TQTempFilter.load(TQTempFilter.get_saved_filters()[0])
    # sf = DOGSpatialFilter.load(DOGSpatialFilter.get_saved_filters()[0])
    # # -----------
    # ===========
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
    # -----------
    # ===========
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
