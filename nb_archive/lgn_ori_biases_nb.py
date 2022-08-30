# +
from typing import Tuple
from scipy.interpolate.interpolate import interp1d
from lif import *
# -
# +
from lif.plot import plot
# -
# +
import plotly.express as px
from scipy.ndimage import gaussian_filter1d
# -
# +
from lif.receptive_field.filters import cv_von_mises as cvvm
# -

# > Poisson Response and Plots
# +
tf = TQTempFilter.load(TQTempFilter.get_saved_filters()[0])
sf = DOGSpatialFilter.load(DOGSpatialFilter.get_saved_filters()[0])
# -
# +
# stim_amp=0.5
spat_res=ArcLength(1, 'mnt')
spat_ext=ArcLength(120, 'mnt')
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

orientation = ArcLength(90, 'deg')
temp_freq = TempFrequency(8)
spat_freq_x = SpatFrequency(2)
spat_freq_y = SpatFrequency(0)
# -
# +
st_params = do.SpaceTimeParams(spat_ext, spat_res, temp_ext, temp_res)
stim_params = do.GratingStimulusParams(
    spat_freq_x, temp_freq,
    orientation=orientation,
    amplitude=1, DC=1
)
# -
# +
resp = conv.mk_single_sf_tf_response(sf, tf, st_params, stim_params)
# -
# +
n_trials = 20
s, pop_s = conv.mk_sf_tf_poisson(st_params, resp, n_trials=n_trials)
# -
# +
all_spikes = conv.aggregate_poisson_trials(s)
# -
# +
plot.poisson_trials_rug(s).show()
# -
# +
plot.psth(st_params, s, 20).show()
# -


# > ori biases
# +
cvvm.kvals
# -
# +
fig = px.line(
    x=cvvm.kvals, y=[cvvm.cvvals, cvvm.k_cv(cvvm.kvals)],
    range_y=[0, 1], labels={'x': 'von mises k', 'value': 'circ var'}
    )
fig.data[0].name = 'actual'
fig.data[1].update(name='interp', mode='markers')
fig.show()
# -
# +
k_val = 0.6
# von_mises max amp is 1, so b is always 1
a, b = (
    cvvm.von_mises(ArcLength(0), k=k_val),
    cvvm.von_mises(ArcLength(np.pi/2, 'rad'), k=k_val)
    )
a, b
# -
# +
target_amp = (a + b)/2
ori_ratio = b/a
target_amp, ori_ratio
# -
# +
x_freqs = SpatFrequency(np.arange(0, 20, 0.1))
spat_resp = ff.mk_dog_sf_ft(x_freqs, SpatFrequency(0), sf.parameters)
angles = ArcLength(np.linspace(0, 180, 8, False))
# -
# +
spat_freqs_x, spat_freqs_y = ff.mk_sf_ft_polar_freqs(
    ArcLength(angles.base - 90), SpatFrequency(15))
# -
# +
spat_resp = ff.mk_dog_sf_ft(spat_freqs_x, spat_freqs_y, sf.parameters)
# -
# +
spat_resp
# -
# +
circ_var = cvvm.circ_var(spat_resp, angles)
circ_var
# -
# +
k_cv(circ_var)
# -
# +
sim_resp = vm(angles.rad, k=k_cv(circ_var), phi=0)
# -
# +
(
    px
    .line_polar(
        r=sim_resp, theta=angles.deg, start_angle=0, direction='counterclockwise'
        )
    .add_trace(
        px.line_polar(r=spat_resp/spat_resp.max(), theta=angles.deg)
        .update_traces(line_color='red', mode='markers').data[0]
        )
    .show())
# -
# +
# original
sf.parameters.cent = Gauss2DSpatFiltParams(
    amplitude=36.4265938532914,
    arguments=Gauss2DSpatFiltArgs(
        h_sd=ArcLength(value=1.4763319256270793, unit='mnt'),
        v_sd=ArcLength(value=1.4763319256270793, unit='mnt')
        )
    )
# -
# +
sf.parameters.cent = Gauss2DSpatFiltParams(
    amplitude=36.4265938532914,
    arguments=Gauss2DSpatFiltArgs(
        h_sd=ArcLength(value=1.8, unit='mnt'),
        v_sd=ArcLength(value=1.2, unit='mnt')
        )
    )
# -
# +
sf.parameters.cent
# -
# +
x_freqs['cpd']
# -
# +
px.line(x=x_freqs.base, y=spat_resp).show()
# -
# +
sf.parameters.cent.arguments.asdict_()
# -
# +
amps = vm(angles.rad, k=0.6)
# -
# +
px.line(x=angles.deg, y=amps).show()
# -


# > Estimate sd vals for given cv
# +
import copy
# -
# +
copy.deepcopy(sf.parameters.cent.arguments)
# -
# +
mk_ori_biased_sd_factors(2)
# -
# +
def mk_ori_biased_sd_factors(ratio: float) -> Tuple[float, float]:
    """Presuming base SD is average, ratio will maintain average of v & h as base

    presumption: a + b = 2, a/b = ratio
    """

    a = (2*ratio)/(ratio+1)
    b = 2 - a

    return a, b

def mk_ori_biased_sf(
        sf: DOGSpatialFilter, v_sd_fact: float, h_sd_fact: float
        ) -> DOGSpatialFilter:
    """Duplicate sf but with v and h sd values multiplied by factors
    """

    args_copy = copy.deepcopy(sf)

    args_copy.parameters.cent.arguments.v_sd = ArcLength(
        args_copy.parameters.cent.arguments.v_sd.mnt * v_sd_fact, 'mnt'
        )
    args_copy.parameters.cent.arguments.h_sd = ArcLength(
        args_copy.parameters.cent.arguments.h_sd.mnt * h_sd_fact, 'mnt'
        )

    return args_copy
# -
# +
new_sf = mk_ori_biased_sf(sf, *mk_ori_biased_sd_factors(1.5))
# -

# at 50% of preferred spat_freq, find h/v ratio -> circ_var
# +
angles = ArcLength(np.linspace(0, 180, 8, False))
# -
# +
spat_freqs_x, spat_freqs_y = ff.mk_sf_ft_polar_freqs(
    ArcLength(angles.base - 90), SpatFrequency(9))
# -
# +
spat_resp = ff.mk_dog_sf_ft(spat_freqs_x, spat_freqs_y, new_sf.parameters)
# -
# +
spat_resp
# -
# +
def orientation_plot(resp, theta):
    resp = np.r_[resp, resp, resp[0]]
    theta = np.r_[theta.deg, theta.deg + 180, theta.deg[0]]
    fig = px.line_polar(r=resp, theta=theta,
            start_angle=0, direction='counterclockwise'
            ).update_traces(mode='lines+markers')
    return fig
# -
# +
orientation_plot(spat_resp, angles).show()
# -
# +
circ_var = cvvm.circ_var(spat_resp, angles)
circ_var
# -
# +
spat_freqs_x, spat_freqs_y = ff.mk_sf_ft_polar_freqs(
    ArcLength(angles.base - 90), SpatFrequency(9))

ratios = np.linspace(1, 50, 1000)

cv_vals = [
    cvvm.circ_var(
        ff.mk_dog_sf_ft(
            spat_freqs_x, spat_freqs_y,
            mk_ori_biased_sf(sf, *mk_ori_biased_sd_factors(ratio)).parameters
            ),
            angles
        )
    for ratio in ratios
]
# -
# +
px.line(x=ratios, y=cv_vals, labels={'x':'ratios', 'y':'circ var'}).show()
# -
# +
spat_resp = ff.mk_dog_sf_ft(
                spat_freqs_x, spat_freqs_y,
                mk_ori_biased_sf(sf, *mk_ori_biased_sd_factors(20)).parameters
            )
orientation_plot(spat_resp, angles).show()
# -
# +
new_sf = mk_ori_biased_sf(sf, *mk_ori_biased_sd_factors(3))
spat_freqs = SpatFrequency(np.linspace(0, 30, 100))

spat_freq_resp_v = ff.mk_dog_sf_ft(
        *ff.mk_sf_ft_polar_freqs(ArcLength(0), spat_freqs),
        new_sf.parameters
    )
spat_freq_resp_h = ff.mk_dog_sf_ft(
        *ff.mk_sf_ft_polar_freqs(ArcLength(-90), spat_freqs),
        new_sf.parameters
    )
# -
# +
fig = px.line(x=spat_freqs.cpd,
        y=[spat_freq_resp_v, spat_freq_resp_h])
fig.data[0].name='vert'
fig.data[1].name='horiz'
fig.show()
# -
# +
v_spat_freq_resp_int = interp1d(spat_freqs.cpd, spat_freq_resp_v)
# -
# +
peak_resp = spat_freq_resp_v.max()
peak_idx = spat_freq_resp_v.argmax()
test = spat_freq_resp_v[peak_idx:] < (0.5 * peak_resp)
# -
test.nonzero()[0][0]
test.argmax()
spat_freqs.cpd[14+39]
test.size, spat_freqs.cpd.size
# +
half_max_resp = 0.5 * spat_freq_resp_v.max()
v_spat_freq_resp_int(half_max_resp)
# -
# +
v_spat_freq_resp_int(10)
# -
# +
def circ_var_sf_ratio_naito_def(
        ratio: float, sf: DOGSpatialFilter,
        angles: ArcLength[np.ndarray], spat_freqs: SpatFrequency[np.ndarray]
        ):

    new_sf = mk_ori_biased_sf(sf, *mk_ori_biased_sd_factors(ratio))

    # spat_freqs = SpatFrequency(np.linspace(0, 30, 100))

    spat_freq_resp_v = ff.mk_dog_sf_ft(
            *ff.mk_sf_ft_polar_freqs(ArcLength(0), spat_freqs),
            new_sf.parameters
        )

    peak_resp = spat_freq_resp_v.max()
    peak_idx = spat_freq_resp_v.argmax()
    threshold_resp_idxs = (spat_freq_resp_v[peak_idx:] < (0.5 * peak_resp))

    # no 50% of max response values
    # return None
    if threshold_resp_idxs.sum() == 0:
        return None

    first_threshold_resp_idx = threshold_resp_idxs.nonzero()[0][0]

    circ_var_spat_freq = spat_freqs.base[peak_idx+first_threshold_resp_idx]

    # angles = ArcLength(np.linspace(0, 180, 8, False) - 90)  # grating drift direction

    spat_freqs_x, spat_freqs_y = ff.mk_sf_ft_polar_freqs(
        angles, SpatFrequency(circ_var_spat_freq))
    resp = ff.mk_dog_sf_ft(spat_freqs_x, spat_freqs_y, new_sf.parameters)
    circ_var = cvvm.circ_var(resp, angles)


    return circ_var
# -
# +
spat_freqs = SpatFrequency(np.linspace(0, 100, 200))
angles = ArcLength(np.linspace(0, 180, 8, False) - 90)  # grating drift direction
# -
# +
circ_var_sf_ratio_naito_def(1.4, sf, angles, spat_freqs)
# -
# +
def naito_cvval(*args):
    try:
        cv = circ_var_sf_ratio_naito_def(*args)
        return cv
    except Exception:
        return None
# -
# +
from functools import partial
from scipy.optimize import minimize_scalar
# -
# +
def find_naito_ratio(sf, circ_var_target, angles, spat_freqs):

    circ_var = partial(circ_var_sf_ratio_naito_def, sf=sf, angles=angles, spat_freqs=spat_freqs)
    def obj_func(ratio):
        cv = circ_var(ratio)
        if cv is None:
            return 1e6
        return abs(cv - circ_var_target)

    res = minimize_scalar(obj_func, method='Bounded', bounds=[1, 50])

    return res
# -
# +
opt_res = find_naito_ratio(sf, 0.15, angles, spat_freqs)
opt_res
# -
# +
circ_var_sf_ratio_naito_def(opt_res.x, sf, angles, spat_freqs)
# -
# +
putative_cv_vals = np.linspace(0, 1, 100)
sf_cv_ratio_opt_ress = [
    find_naito_ratio(sf, cv, angles, spat_freqs) for cv in putative_cv_vals
    ]
sf_cv_ratio_vals = [
    opt_res.x if opt_res.success else None for opt_res in sf_cv_ratio_opt_ress
    ]
# -
# +
px.line(x=sf_cv_ratio_vals, y=putative_cv_vals).show()
# -
# +
cv_ratio_interp = interp1d(putative_cv_vals, sf_cv_ratio_vals)
px.line(
    x=putative_cv_vals, y=cv_ratio_interp(putative_cv_vals),
    labels={'x': 'circ_var', 'y':'sd ratio'}
    ).show()
# -
%debug
# +
naito_cvvals = np.array([naito_cvval(ratio, sf, angles, spat_freqs) for ratio in ratios])
# -
np.isnan(naito_cvvals)


# +
dir(sf.ori_bias_params)
# -


# > Testing First Ori Biases Implementation

# +
from scipy.interpolate.interpolate import interp1d
from lif.utils.units.units import ArcLength, SpatFrequency, TempFrequency, Time
from lif.utils import data_objects as do
from lif.receptive_field.filters import (
    filter_functions as ff, filters)
# -
# >> clean up temp data from tracey (Kaplan 1987)
# +
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as psp
# -

# >> Data directory
# +
import lif.utils.settings as settings
data_dir = settings.get_data_dir()
# -
# +
saved_sfs = do.DOGSpatialFilter.get_saved_filters()
# saved_sfs
# -
# +
sf = do.DOGSpatialFilter.load(saved_sfs[0])
# -


# > Ori Plot Proto
# +
from importlib import reload
reload(ff)
# -
# +
sf.parameters.mk_ori_biased_duplicate(*ff.mk_ori_biased_sd_factors(0.5))
# -
# +
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
        spat_freq: SpatFrequency[float], sf_params: do.DOGSpatFiltArgs,
        n_angles: int = 8):


    angles, ori_resp = mk_sf_orientation_resp(spat_freq, sf_params, n_angles)

    # wrap
    resp = np.r_[ori_resp, ori_resp, ori_resp[0]]
    theta = np.r_[angles.deg, angles.deg + 180, angles.deg[0]]

    fig = (
            px
            .line_polar(
                r=resp, theta=theta,
                start_angle=0, direction='counterclockwise')
            .update_traces(mode='lines+markers', fill='toself')
        )
    return fig
# -
# +
def ori_spat_freq_heatmap(
        sf_params: do.DOGSpatFiltArgs,
        n_orientations: int = 8,
        n_spat_freqs: int = 50,
        width: int = 500, height: int = 500):

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
    fig.update_xaxes(tickvals=angles.deg, ticks='outside')

    return fig
# -
# +
ori_spat_freq_heatmap(sf.parameters).show()
# -
# +
ori_spat_freq_heatmap(sf.mk_ori_biased_parameters(0.5), n_orientations=32).show()
# -
# +
orientation_plot(
    SpatFrequency(10),
    sf.mk_ori_biased_parameters(0.2)
    ).show()
# -

# >> Spat Freq and Circ Var Subplots
# +
def orientation_circ_var_subplots(
        sf_params: do.DOGSpatFiltArgs,
        spat_freq_factors: np.ndarray = np.array([1, 2, 4, 8]),
        circ_vars: np.ndarray = np.arange(0.1, 0.6, 0.1)
        ):


    spat_freqs = SpatFrequency(
        np.linspace(
            0,
            ff.find_null_high_sf(sf_params).base,
            100)
        )
    sf_resp = ff.mk_dog_sf_ft(
        *ff.mk_sf_ft_polar_freqs(ArcLength(90), spat_freqs),
        sf_params)

    max_resp_idx = sf_resp.argmax()
    max_spat_freq = spat_freqs.base[max_resp_idx]
    spat_freqs = SpatFrequency(
        max_spat_freq * spat_freq_factors  # type: ignore
        )

    n_cols = spat_freqs.base.size
    n_rows = circ_vars.size

    fig = psp.make_subplots(
        rows=n_rows, cols=n_cols,
        start_cell='bottom-left',
        specs=[[{'type':'polar'}]*n_cols]*n_rows,
        column_titles= [
            (
                f'{round(s, 2)}cpd ({fact}X)'
                if s != max_spat_freq
                else f'<b>{round(s,2)}cpd (pref)</b>'
            )
            for s, fact in zip(spat_freqs.cpd, spat_freq_factors)
            ],
        x_title="Spat Freq", y_title="Circ Var",
        row_titles= [f'{round(cv, 2)}' for cv in circ_vars],
        )
    fig = fig.update_polars(radialaxis_nticks=2, angularaxis_showticklabels=False)  # type: ignore

    # add subplots
    for sfi, spat_f in enumerate(spat_freqs.base):
        for cvi, cv in enumerate(circ_vars):
            ori_fig = orientation_plot(
                do.SpatFrequency(spat_f),
                sf.mk_ori_biased_parameters(cv)
                )
            fig.add_trace(ori_fig.data[0], col=sfi+1, row=cvi+1)


    return fig

# -


# +
fig = orientation_circ_var_subplots(sf.parameters)
# -
# +
fig.show()
# -
# +
# >>> Make Spat Tuning Curve
spat_freqs = SpatFrequency(np.linspace(0, ff.find_null_high_sf(sf.parameters).base, 100))
sf_resp = ff.mk_dog_sf_ft(*ff.mk_sf_ft_polar_freqs(ArcLength(90), spat_freqs), sf.parameters)
px.line(x=spat_freqs.cpd, y=sf_resp).show()
# -
# +
# >>> Estimate Preferred Spat Freq and Multiples and Circ Vars
max_resp_idx = sf_resp.argmax()
max_spat_freq = spat_freqs.base[max_resp_idx]
spat_freq_factors = np.array([1, 1.5, 2, 3, 4, 8])
spat_freqs = SpatFrequency(
    # octaves of preferred spat_freq from 0.5 to 8
    max_spat_freq * spat_freq_factors  # type: ignore
    )
circ_vars = np.arange(0.1, 0.6, 0.1)
# -
# +
# >>> Subplots
import plotly.subplots as psp
# -
# +
n_cols = spat_freqs.base.size
n_rows = circ_vars.size
fig = psp.make_subplots(
    rows=n_rows, cols=n_cols,
    start_cell='bottom-left',
    specs=[[{'type':'polar'}]*n_cols]*n_rows,
    column_titles= [
        f'{round(s, 2)}cpd ({fact}X)' if s != max_spat_freq else f'<b>{round(s,2)}cpd (pref)</b>'
        for s, fact in zip(spat_freqs.cpd, spat_freq_factors)
        ],
    x_title="Spat Freq", y_title="Circ Var",
    row_titles= [f'{round(cv, 2)}' for cv in circ_vars],
    )
fig.update_polars(radialaxis_nticks=2, angularaxis_showticklabels=False);

for sfi, spat_f in enumerate(spat_freqs.base):
    for cvi, cv in enumerate(circ_vars):
        ori_fig = orientation_plot(do.SpatFrequency(spat_f), sf.mk_ori_biased_parameters(cv))
        fig.add_trace(ori_fig.data[0], col=sfi+1, row=cvi+1)

# -
# +
fig.show()
# -
# +
fig.write_image('sf_cv_sp.pdf', width=1200, height=800)
# -
# +
# factor by octaves (0.5, 1, 2, 4, 8)
# then circ vars (0, 0.1, 0.2, 0.3, 0.5) (??)
# -

# >> Testing Codification

# +
from importlib import reload
reload(plot)
reload(ff)
# -
# +
plot.orientation_plot(
    ff.mk_ori_biased_spatfilt_params_from_spat_filt(sf, 0.3),
    SpatFrequency(7)
    ).show()
# -
# +
plot.ori_spat_freq_heatmap(
    ff.mk_ori_biased_spatfilt_params_from_spat_filt(sf, 0.3),
    n_orientations=16
    ).show()
# -
# +
plot.orientation_circ_var_subplots(sf).show()
# -

# XX make heatmap of the orientation response to various spatial freqs and v (sf v orientation)
# XX make polar orientation response for a few spatial freqs
# XX codify into plotting module
# write some tests (don't know what exactly ... maybe for range of sds, can produce ori biases that return response of circ_var desired)

