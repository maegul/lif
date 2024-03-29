from typing import Optional, Tuple, Union, Dict, Sequence, Literal

import brian2
from scipy.ndimage.filters import gaussian_filter1d

from ..receptive_field.filters import (
    filters,
    filter_functions as ff,
    cv_von_mises as cvvm)
from ..receptive_field.filters.filter_functions import (
    do, ArcLength, SpatFrequency, Time, TempFrequency, scalar)
from ..convolution import estimate_real_amp_from_f1 as est_amp
# from lif.receptive_field.filters.data_objects import DOGSpatialFilter, TQTempFilter
from ..convolution import convolve, correction
from ..utils.data_objects import DOGSpatialFilter, SpaceTimeParams, TQTempFilter

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

# # Utility Color Objects


spat_filt_colors = {
        key: px.colors.qualitative.Dark24[i]
        for i, key in enumerate(filters.spatial_filters.keys())
    }

# > Filters

def tq_temp_filt(
        temp_filter: do.TQTempFilter,
        temp_res: Time = Time(1, 'ms'), temp_ext: Optional[Time] = None,
        tau: Union[bool, float] = False):

    # tau arg when bool is whether to use tau with the default factor
    # when float, it is the above and the factor simultaneously ... for quick interface
    if (temp_ext and tau) or ((not temp_ext) and (not tau)):
        raise ValueError('Must provide only one of temp_ext or tau')

    filter_tau = temp_filter.parameters.arguments.tau

    # kinda not ok ... but maybe(?) worth it for the simple interface of this function ...
    # now just need to provide either temp_ext or tau args.
    if temp_ext:
        temp_coords = ff.mk_temp_coords(temp_res, temp_ext)
    else:
        temp_coords = ff.mk_temp_coords(
            temp_res,
            tau=filter_tau,
            temp_ext_n_tau=(tau if not isinstance(tau, bool) else None)
            )

    filter_values = ff.mk_tq_tf(temp_coords, temp_filter.parameters)

    fig = (
        go.Figure()
        .add_trace(
            go.Scatter(
                name='Temp Filter',
                x=temp_coords.ms, y=filter_values
                )
            )
        .update_layout(
            xaxis_title='Time (ms)', yaxis_title='Amplitude')
    )

    return fig


def _mk_spat_filt_values(
        spat_filter: Union[do.DOGSpatialFilter,do.DOGSpatFiltArgs],
        spat_res: ArcLength[scalar] = ArcLength(1, 'mnt'),
        spat_ext: Optional[ArcLength[scalar]] = None,
        sd: Union[bool, float] = False,
        ) -> Tuple[ArcLength[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:

    if (spat_ext and sd) or ((not spat_ext) and (not sd)):
        raise ValueError('Must provide only one of spat_ext and sd')

    filter_sd = spat_filter.parameters.max_sd()

    if spat_ext:
        spat_coords = ff.mk_sd_limited_spat_coords(
            spat_res=spat_res, spat_ext=spat_ext )

    else:
        spat_coords = ff.mk_sd_limited_spat_coords(
            spat_res=spat_res,
            sd=filter_sd,
            sd_limit=(sd if (not isinstance(sd, bool)) else None)
            )

    cent_params = spat_filter.parameters.cent
    surr_params = spat_filter.parameters.surr

    # #! This will produce excessive amplitudes
    #       as in 2D version, the vectors are multiplied by each other
    #       ... and tend to have values <1, and so reduce amplitude overall
    cent_filter_values = cent_params.amplitude * ff.mk_gauss_1d(
        coords=spat_coords, sd=cent_params.arguments.h_sd)
    surr_filter_values = -1 * surr_params.amplitude * ff.mk_gauss_1d(
        coords=spat_coords, sd=surr_params.arguments.h_sd)
    sf_values = cent_filter_values + surr_filter_values

    return spat_coords, cent_filter_values, surr_filter_values, sf_values


def spat_filt(
        spat_filter: Union[do.DOGSpatialFilter,do.DOGSpatFiltArgs],
        spat_res: ArcLength[scalar] = ArcLength(1, 'mnt'),
        spat_ext: Optional[ArcLength[scalar]] = None,
        sd: Union[bool, float] = False
        ):

    spat_coords, cent_filter_values, surr_filter_values, sf_values = _mk_spat_filt_values(
        spat_filter, spat_res, spat_ext, sd)

    fig = (
        go.Figure()
        .add_trace(
            go.Scatter(
                name='cent',
                x=spat_coords.mnt, y=cent_filter_values,
                line={'color': 'red'}
                )
            )
        .add_trace(
            go.Scatter(
                name='surr',
                x=spat_coords.mnt, y=surr_filter_values,
                line={'color': 'blue'}
                )
            )
        .add_trace(
            go.Scatter(
                name='RF',
                x=spat_coords.mnt, y=sf_values,
                line={'color': 'purple', 'width': 4}
                )
            )
        .update_layout(
            xaxis_title='Distance (arc mnts)', yaxis_title='Amplitude')
        .update_yaxes(zeroline=True, zerolinewidth=5, zerolinecolor='grey')  # type: ignore
        )

    return fig


def multi_spat_filt(
        spat_filters: Sequence[do.DOGSpatialFilter],
        spat_res: ArcLength[scalar] = ArcLength(1, 'mnt'),
        spat_ext: Optional[ArcLength[scalar]] = None,
        sd: Union[bool, float] = False,
        normalise_magnitude: bool = False
        ):

    spat_filter_data = []
    for sf in spat_filters:
        spat_coords, _, _, sf_values = (
            _mk_spat_filt_values(sf, spat_res, spat_ext, sd)
            )
        label = f'{sf.source_data.meta_data.author} ({sf.source_data.meta_data.year}), {sf.source_data.meta_data.reference}'
        spat_filter_data.append((label, spat_coords, sf_values, sf_values/sf_values.max()))

    fig = go.Figure()

    for sf_data in spat_filter_data:
        label, spat_coords, values, values_norm = sf_data
        fig = fig.add_scatter(  # type: ignore
                name = label,
                x=spat_coords.mnt,
                y=(
                    values
                        if not normalise_magnitude
                        else
                        values_norm
                    )
            )

    fig = (
        fig
        .update_layout(
            xaxis_title='Distance (arc mnts)', yaxis_title='Amplitude')
        .update_yaxes(zeroline=True, zerolinewidth=5, zerolinecolor='grey')  # type: ignore
    )

    return fig

# > Filter Fits

def tq_temp_filt_fit(
        fit_filter: do.TQTempFilter,
        n_temp_freqs: int = 50,
        use_log_xaxis: bool = True, use_log_freqs: bool = True
        ):

    # render the filter in frequency domain (as the original data)
    freqs = fit_filter.source_data.data.frequencies
    tf_ft = ff.mk_tq_tf_ft(freqs=freqs, tf_params=fit_filter.parameters)

    # render high-resolution filter in the frequency domain
    min_freq, max_freq = freqs.base.min(), freqs.base.max()
    if use_log_freqs:
        min_exp, max_exp = (np.log10(v) for v in (min_freq, max_freq))
        freqs_hres = TempFrequency(10**np.linspace(min_exp, max_exp, n_temp_freqs))
    else:
        freqs_hres = TempFrequency(np.linspace(min_freq, max_freq, n_temp_freqs))

    tf_ft_hres = ff.mk_tq_tf_ft(freqs=freqs_hres, tf_params=fit_filter.parameters)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name='tq_filter_hres',
            x=freqs_hres.hz, y=tf_ft_hres,
            mode='lines', line={'color':'gray', 'dash':'3px,1px'}
            )
        )
    fig.add_trace(
        go.Scatter(
            name='tq_filter',
            x=freqs.hz,
            y=tf_ft,
            mode='lines+markers'
            )
        )
    fig.add_trace(
        go.Scatter(
            name='source',
            x=freqs.hz,
            y=fit_filter.source_data.data.amplitudes,
            mode='lines+markers'
            )
        )
    fig.update_layout(xaxis_title='Frequency (Hz)', yaxis_title='Amplitude')

    if use_log_xaxis:
        fig = fig.update_xaxes(type='log')  # type: ignore

    return fig


def spat_filt_fit(
        fit_filter: do.DOGSpatialFilter,
        n_spat_freqs: int = 50,
        normalise_magnitude: bool = False,
        use_log_xaxis: bool = True, use_log_freqs: bool = True,
        hi_res_fit_only: bool = False,
        min_freq_bound: Optional[SpatFrequency[scalar]] = None,
        max_freq_bound: Optional[SpatFrequency[scalar]] = None
        ):

    # render the filter in frequency domain (as the original data)
    freqs = fit_filter.source_data.data.frequencies
    sf_ft = ff.mk_dog_sf_ft_1d(freqs=freqs, dog_args=fit_filter.parameters)

    # render high-resolution filter in the frequency domain
    if min_freq_bound is None or max_freq_bound is None:
        min_freq, max_freq = freqs.base.min(), freqs.base.max()
    elif (min_freq_bound is None or max_freq_bound is None) and not (min_freq_bound and max_freq_bound):
        raise ValueError('Must provide both min and max freq')
    else:
        min_freq, max_freq = min_freq_bound.base, max_freq_bound.base

    if use_log_freqs:
        min_exp, max_exp = (np.log10(v) for v in (min_freq, max_freq))
        freqs_hres = SpatFrequency(10**np.linspace(min_exp, max_exp, n_spat_freqs))
    else:
        freqs_hres = SpatFrequency(np.linspace(min_freq, max_freq, n_spat_freqs))

    sf_ft_hres = ff.mk_dog_sf_ft_1d(freqs=freqs_hres, dog_args=fit_filter.parameters)
    source_ft = fit_filter.source_data.data.amplitudes

    if normalise_magnitude:
        norm_factor = sf_ft_hres.max()
        sf_ft_hres = sf_ft_hres / norm_factor
        sf_ft = sf_ft / norm_factor
        source_ft = source_ft / norm_factor

    fig = go.Figure()

    if hi_res_fit_only:
        fig.add_trace(
            go.Scatter(
                name='spat_filter_hres',
                x=freqs_hres.cpd, y=sf_ft_hres,
                line={'dash': '3px,1px'})
            )
    else:
        fig.add_trace(
            go.Scatter(
                name='spat_filter_hres',
                x=freqs_hres.cpd, y=sf_ft_hres,
                line={'color': 'grey', 'dash': '3px,1px'})
            )
        fig.add_trace(
            go.Scatter(
                name='spat_filter',
                x=freqs.cpd, y=sf_ft)
            )
        fig.add_trace(
            go.Scatter(
                name='source',
                x=freqs.cpd, y=source_ft)
            )

    fig.update_layout(
        xaxis_title='Spatial Freq (CPD)',
        yaxis_title='Amplitude' if not normalise_magnitude else 'Amplitude (normalised to hres)')

    if use_log_xaxis:
        fig.update_layout(xaxis_type='log')

    return fig


def multi_spat_filt_fit(
        fit_filters: Sequence[do.DOGSpatialFilter],
        n_spat_freqs: int = 50,
        normalise_magnitude: bool = True,
        use_log_xaxis: bool = True, use_log_freqs: bool = True,
        share_freq_bounds: bool = False,
        ):

    # Calculate all min and max freqs
    if share_freq_bounds:
        collective_min_freq = SpatFrequency(
                min(
                    fit_filter.source_data.data.frequencies.base.min()
                    for fit_filter in fit_filters
                )
            )
        collective_max_freq = SpatFrequency(
                max(
                    fit_filter.source_data.data.frequencies.base.max()
                    for fit_filter in fit_filters
                )
            )
    else:
        # reset to values expected in lower function
        collective_min_freq, collective_max_freq = None, None

    fig = go.Figure()

    for sf in fit_filters:
        index_key = filters.reverse_spatial_filters[sf.key]
        new_graph = spat_filt_fit(sf, n_spat_freqs,
            normalise_magnitude,
            use_log_xaxis, use_log_freqs,
            hi_res_fit_only=True,
            min_freq_bound=collective_min_freq, max_freq_bound=collective_max_freq
            )
        fig.add_trace(
            new_graph.update_traces(
                    name=index_key,
                    line={'dash': 'solid'}
                ).data[0]
            )

    if use_log_xaxis:
        fig.update_layout(xaxis_type='log')

    fig.update_layout(
        xaxis_title='Spatial Freq (CPD)',
        yaxis_title='Amplitude' if not normalise_magnitude else 'Amplitude (normalised to hres)')

    return fig


def lgn_cell_locations(lgn_layer: do.LGNLayer):

    x_locs = [c.location.x.mnt for c in lgn_layer.cells]
    y_locs = [c.location.y.mnt for c in lgn_layer.cells]

    fig = (
        px
        .scatter(x=x_locs, y=y_locs)
        .update_yaxes(scaleanchor = "x", scaleratio = 1)
        .update_layout(xaxis_constrain='domain', yaxis_constrain='domain')
        )

    return fig


def mk_ellipse_arc(
        x_center: float=0, y_center: float=0, a: float=1., b: float=1.,
        ori: ArcLength[scalar] = ArcLength(0), N=20
        ) -> Tuple[np.ndarray, np.ndarray]:

    phi = np.linspace(0, np.pi*2, N)
    theta = ori.rad
    x = x_center + a*np.cos(phi)*np.cos(theta) - b*np.sin(phi)*np.sin(theta)
    y = y_center + a*np.cos(phi)*np.sin(theta) + b*np.sin(phi)*np.cos(theta)

    return x, y


def lgn_sf_locations_and_shapes(
    lgn_layer: do.LGNLayer,
    showlegend: bool = False,
    highlight_idxs: Optional[Sequence[int]] = None,
    color_ellipses: bool = False,
    coords_at_magnitude: Optional[dict] = None
    ):

    highlight_idxs_set = set(highlight_idxs) if highlight_idxs else set()  # empty set works for lookup

    fig = (
        go.Figure()
        .update_yaxes(scaleanchor = "x", scaleratio = 1)  # type: ignore
        .update_layout(xaxis_constrain='domain', yaxis_constrain='domain', showlegend=showlegend)
        )
    for i, c in enumerate(lgn_layer.cells):
        x_loc = c.location.x.mnt
        y_loc = c.location.y.mnt
        key = filters.reverse_spatial_filters[c.spat_filt.key]

        if i in highlight_idxs_set:
            color='red'
            line_color=color  # for ellipses
            size=14
        else:
            color=spat_filt_colors[key]
            line_color = color if color_ellipses else 'black'
            size=10

        fig.add_scatter(
            x=[x_loc], y=[y_loc],
            mode='markers',
            hovertext=key,
            marker_color=color, marker_size=size,
            name=key
            )

        if coords_at_magnitude:
            # infer shape and dimensions of major/minor axes from ratio of oriented/circular
            h_sd_factor = (
                c.oriented_spat_filt_params.cent.arguments.h_sd.base /
                c.spat_filt.parameters.cent.arguments.h_sd.base
                )
            v_sd_factor = (
                c.oriented_spat_filt_params.cent.arguments.v_sd.base /
                c.spat_filt.parameters.cent.arguments.v_sd.base
                )

            h_sd, v_sd = (
                coords_at_magnitude[c.spat_filt.key].mnt * h_sd_factor,
                coords_at_magnitude[c.spat_filt.key].mnt * v_sd_factor,
                )
        else:
            # if no coords provided, revert simply to 2sd of center
            h_sd, v_sd = (
                2 * c.oriented_spat_filt_params.parameters.cent.arguments.h_sd.mnt,
                2 * c.oriented_spat_filt_params.parameters.cent.arguments.v_sd.mnt
                )
        x_sf, y_sf = mk_ellipse_arc(x_loc, y_loc, a=h_sd, b=v_sd, ori=c.orientation)

        fig.add_scatter(
            x=x_sf, y=y_sf,
            mode='lines',
            hovertext=key,
            line_color=line_color,
            name=key
            )

    return fig

    # x_locs = [c.location.x.mnt for c in lgn_layer.cells]
    # y_locs = [c.location.y.mnt for c in lgn_layer.cells]
    # keys = [filters.reverse_spatial_filters[c.spat_filt.key] for c in lgn_layer.cells]

    # if highlight_idxs:
    #     highlight_idxs_set = set(highlight_idxs)
    #     colors = [
    #         spat_filt_colors[k] if not (i in highlight_idxs_set) else
    #             'red'
    #             for i, k in enumerate(keys)
    #         ]
    #     sizes = [
    #         10 if not (i in highlight_idxs_set) else
    #             14
    #         for i, _ in enumerate(keys)
    #         ]
    # else:
    #     colors = [spat_filt_colors[k] for k in keys]
    #     sizes = [10 for _ in lgn_layer.cells]

    # fig=go.Figure()
    # fig = (
    #     fig
    #     .add_scatter(
    #         x=x_locs, y=y_locs,
    #         hovertext=keys,
    #         mode='markers',
    #         marker_color=colors,
    #         marker_size=sizes
    #         )
    #     .update_yaxes(scaleanchor = "x", scaleratio = 1)
    #     .update_layout(xaxis_constrain='domain', yaxis_constrain='domain')
    #     )

    # for i, c in enumerate(lgn_layer.cells):

    #     rf_line_color='black'

    #     x_loc, y_loc = c.location.x.mnt, c.location.y.mnt
    #     h_sd, v_sd = (
    #         c.oriented_spat_filt_params.parameters.cent.arguments.h_sd.mnt,
    #         c.oriented_spat_filt_params.parameters.cent.arguments.v_sd.mnt
    #         # c.spat_filt.parameters.cent.arguments.h_sd.mnt,
    #         # c.spat_filt.parameters.cent.arguments.v_sd.mnt
    #         )

    #     x0=x_loc-(2*h_sd)
    #     x1=x_loc+(2*h_sd)
    #     y0=y_loc-(2*v_sd)
    #     y1=y_loc+(2*v_sd)

    #     fig.add_shape(
    #         type="circle",
    #         xref="x", yref="y",
    #         x0=x0, x1=x1,
    #         y0=y0, y1=y1,
    #         line_color=rf_line_color,
    #         )

    # return fig


def tq_temp_filt_profile(
        fit_filter: do.TQTempFilter,
        temp_res: Time = Time(1, 'ms'), temp_ext: Time = Time(200, 'ms'),
        **kwargs
        ):
    """Plot a fitted tq temp filter against the original data
    """

    # render the filter in the time domain
    time = ff.mk_temp_coords(temp_res=temp_res, temp_ext=temp_ext)
    tf = ff.mk_tq_tf(t=time, tf_params=fit_filter.parameters)
    # # render the filter in frequency domain (as the original data)
    # freqs = fit_filter.source_data.data.frequencies
    # tf_ft = ff.mk_tq_tf_ft(freqs=freqs, tf_params=fit_filter.parameters)

    # filt = px.line(x=time.ms, y=tf)
    filt = go.Scatter(x=time.ms, y=tf, mode='lines', showlegend=False)
    filt_ft = tq_temp_filt_fit(fit_filter, **kwargs)


    fig = (
        make_subplots(rows=1, cols=2,
            subplot_titles=[
                'Filter (time domain)',
                'Filter and data (frequency domain)'],
            y_title='Amplitude'
            )
        .add_trace(filt, 1, 1)
        .add_traces(filt_ft.data, 1, 2)
        .update_xaxes(title_text='Time (ms)', row=1, col=1)
        .update_xaxes(title_text='Frequency (Hz)', type='log', row=1, col=2)
        )

    return fig


# > Oriented Spatial Filters

def von_mises_k_circ_var():
    """Convenience function to show relationship between von mises and circ var

    Data is precalculated in the cv_von_mises module and used directly from there

    The "interpolated" line is from linear interpolation.
    """
    fig = (px
        .line(
            x=cvvm.kvals, y=[cvvm.cvvals, cvvm.k_cv(cvvm.kvals)],
            range_y=[0, 1],
            labels={'x': 'von mises k', 'value': 'circ var'})
        )

    fig.data[0].name = 'actual'
    fig.data[1].update(name='interpolated', mode='markers')

    return fig


def dog_sf_ft_hv(
        dog_args: do.DOGSpatFiltArgs,
        n_spat_freqs: int = 50,
        use_log_freqs: bool = True,
        freqs: Optional[SpatFrequency] = None,
        use_log_xaxis: bool = True,
        width: int = 800, height: int = 400
        ):
    '''Response of DOG spat filter to stimuli oriented vertically and horizontally

    Uses fourier transform functions to do so analytically.

    Args:
        dog_args: Arguments/parameters of spatial filter (ideally orientation biased)
        n_spat_freqs: For how many spatial frequencies to get the response.
                        Will automatically range from zero (or one) to the highest that elicits
                        a response from the RF
        use_log_freqs: Automatically range frequencies in equal `logarithmic(10)` steps.
                        In this case, start from `10^-1` not `0`.
        freqs: Override the automatic calculation of frequencies and instead provide
                them manually
        use_log_xaxis: Plot the xaxis in logarithmic format (ie, factors of 10 equally spaced)

    Returns:
        plotly figure object
    '''

    # if no specific freqs are provided, generate automatically
    if not freqs:
        max_spat_freq = cvvm.find_null_high_sf(dog_args)
        if use_log_freqs:
            max_freq_exponent = np.log10(max_spat_freq.base)
            freq_exponents = np.linspace(-1, max_freq_exponent, n_spat_freqs)
            freqs = do.SpatFrequency(10**freq_exponents)
        else:
            freqs = do.SpatFrequency(np.linspace(0, max_spat_freq.base, n_spat_freqs))


    # angle of modulation (90-->horizontal) -----------V
    x_freq, y_freq = ff.mk_sf_ft_polar_freqs(ArcLength(90, 'deg'), freqs)
    resp_h = ff.mk_dog_sf_ft(freqs_x=x_freq, freqs_y=y_freq, dog_args=dog_args)

    # angle of modulation (0-->vertical) --------------V
    x_freq, y_freq = ff.mk_sf_ft_polar_freqs(ArcLength(0, 'deg'), freqs)
    resp_v = ff.mk_dog_sf_ft(freqs_x=x_freq, freqs_y=y_freq, dog_args=dog_args)

    fig = (
        px
        .line(x=freqs.cpd, y=resp_h,
            labels={'x': 'Spat Freq (CPD)', 'y': 'Response'},
            width=width, height=height)
        .update_traces(name='0deg ori =', showlegend=True)
        .add_trace(px.line(x=freqs.cpd, y=resp_v)
            .update_traces(
                line_color='red',
                name='90deg ori ||',
                showlegend=True
                ).data[0]
            )
    )

    if use_log_xaxis:
        fig = fig.update_xaxes(type='log')

    return fig


def mk_sf_orientation_resp(
        spat_freq: SpatFrequency[float], sf_params: do.DOGSpatFiltArgs,
        n_angles: int = 8
        ) -> Tuple[ArcLength[np.ndarray], np.ndarray]:
    """Generates orientation response using spat filt ft methods

    returns orientations used and response array

    orientations are adjusted to represent orientation of stimulus, not direction
    of modulation.  Also wrapped to be within [0, 180].
    """

    angles = cvvm.mk_even_semi_circle_angles(n_angles)
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
        use_log_freqs: bool = True, use_log_yaxis: bool = True,
        log_yaxis_start_exponent: float = 0,
        width: int = 500, height: int = 500):
    """Heatmap of with ori on x and spat_freq on y, color ~ response

    """

    angles = cvvm.mk_even_semi_circle_angles(n_orientations)

    max_spat_freq = cvvm.find_null_high_sf(sf_params)
    if use_log_freqs:
        max_freq_exponent = np.log10(max_spat_freq.base)
        freq_exponents = np.linspace(log_yaxis_start_exponent, max_freq_exponent, n_spat_freqs)
        spat_freqs = do.SpatFrequency(10**freq_exponents)
    else:
        spat_freqs = do.SpatFrequency(np.linspace(0, max_spat_freq.base, n_spat_freqs))

    resps = np.empty((spat_freqs.base.size, angles.base.size))
    for i, ori in enumerate(angles.deg):
        spat_freqs_x, spat_freqs_y = ff.mk_sf_ft_polar_freqs(ArcLength(ori, 'deg'), spat_freqs)
        resps[:, i] = ff.mk_dog_sf_ft(spat_freqs_x, spat_freqs_y, sf_params)

    fig = px.imshow(
            resps, x=angles.deg, y=spat_freqs.cpd,
            labels={'x': 'Orientation (Deg)', 'y': 'Spat Freq (CPD)'},
            origin='lower', width=width, height=height, aspect='auto')
    fig = fig.update_xaxes(tickvals=angles.deg, ticks='outside')  # type: ignore

    if use_log_yaxis:
        fig = fig.update_yaxes(type='log')

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
            cvvm.find_null_high_sf(sf.parameters).base,
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
            ori_sf_params = ff.mk_ori_biased_spatfilt_params_from_spat_filt(sf, cv)
            ori_fig = orientation_plot(
                ori_sf_params,
                do.SpatFrequency(spat_f)
                )
            fig.add_trace(ori_fig.data[0], col=sfi+1, row=cvi+1)

    return fig


def circ_var_sd_ratio_method_comparison(
        sf: do.DOGSpatialFilter,
        ratios: Optional[np.ndarray] = None):
    """For provided spatial filter, plot circular variance over sd ratio
    for all the available methods.
    """

    if not ratios:
        ratios = np.linspace(1, 10, 100)

    all_methods = sf.ori_bias_params.all_methods()

    fig = (
        go.Figure()
        .add_traces(
            [
                go.Scatter(
                        name=f'{method.title()} method',
                        x = ratios,
                        y = sf.ori_bias_params.ratio2circ_var(ratios, method=method) )
                    for method in all_methods
            ]
        )
        .update_xaxes(title='SD Ratio')
        .update_yaxes(title='Circular Variance')
        .update_layout(title='SD Ratio to Circular Variance')
        )

    return fig

# > !! stimuli

# old code ... hopefully still works!
# need to install pyqtgraph ... which should be trivial

# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui, 'rad'

def view_stim(stim, axes={'y': 0, 'x': 1, 't': 2}, xvals=None):
    #     if time_axis == 2:
    #         stim = stim.swapaxes(0,2).swapaxes(1,2)
    #     if time_axis == 1:
    #         stim = stim.swapaxes(0,1)

    if not xvals:
        xvals = np.arange(stim.shape[axes['t']])
    pg.image(stim, xvals=xvals, axes=axes)
    QtGui.QApplication.instance().exec_()

# > Convolution and LIF results

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
        n_increments: int = 20, width=650, height=650,
        contrast: Optional[do.ContrastValue] = None
        ):
    '''Plots joint distribution of Response Amplitudes to spatial and temporal frequencies

    Also plots original temporal and spatial frequency response data the filters
    were based on with the "*intersection*" as solid lines.

    Joint response from joint_spat_temp_conv_amp()
    '''

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

    joint_sf_tf_amp = correction.joint_spat_temp_f1_magnitude(
        temp_freqs=tf_freq_mg, spat_freqs_x=sf_freq_mg_x, spat_freqs_y=sf_freq_mg_y,
        sf=sf, tf=tf,
        contrast=contrast
        )

    main_size, margin_size = 0.7, 0.3
    fig = make_subplots(
        2, 2,
        shared_xaxes=True, shared_yaxes=True,
        column_widths=[main_size, margin_size], row_heights=[margin_size, main_size])

    joint_heatmap = (
        px
        .imshow(
            joint_sf_tf_amp, x=spat_freqs_x.cpd, y=temp_freqs.hz, origin='lower',
            labels={'x': 'SF', 'y': 'TF', 'color': 'Amp'}
            )
    )

    base_sf, base_tf = (
        tf.source_data.resp_params.sf.cpd, sf.source_data.resp_params.tf.hz
        )

    sfp = spat_filt_fit(sf, use_log_xaxis=False, use_log_freqs=False)
    tfp = tq_temp_filt_fit(tf, use_log_xaxis=False, use_log_freqs=False)

    # swap x an y for temp filt to align with the heatmap
    for tr in tfp.data:
        tr.x, tr.y = tr.y, tr.x  # type: ignore

    plot = (
        fig
        .add_traces(
            #                       V--all traces in sfp ("splat" operator)
            [joint_heatmap.data[0], *sfp.data, *tfp.data],
            #   V--splat for "1" repeated for every sfp trace
            rows=[2, *([1]*len(sfp.data)), *([2]*len(tfp.data))],  # type: ignore
            cols=[1, *([1]*len(sfp.data)), *([2]*len(tfp.data))])  # type: ignore
        .update_xaxes(title='Spat Freq (CPD)', row=2, col=1)
        .update_yaxes(title='Temp Freq (Hz)', row=2, col=1)
        .add_vline(
            x=base_sf, row=1, col=1, annotation_text=f'{base_sf} CPD',
            annotation_position='bottom right')
        .add_hline(
            y=base_tf, row=2, col=2, annotation_text=f'{base_tf} Hz',
            annotation_position='top left')
        .update_layout(width=width, height=height,
            coloraxis_colorbar={'len': main_size, 'yanchor': 'bottom', 'y': 0}
            )
    )

    return plot


def poisson_trials_rug(spike_data: Union[Dict, np.ndarray, brian2.SpikeMonitor]):
    """Plot rug plot for each spike for each trial

    """

    all_spikes = convolve.aggregate_poisson_trials(spike_data)

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


