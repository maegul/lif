
# Testing the current rf locations system.

# Does it actually produce good pairwise distance distributions ... can it be improved??


# # Imports
# +
from pathlib import Path
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from lif.utils.units.units import (
        ArcLength, Time, SpatFrequency, TempFrequency
    )
import lif.utils.data_objects as do
import lif.utils.settings as settings

from lif.receptive_field.filters import filters
import lif.lgn as lgn
import lif.lgn.rf_locations as rf_locs
from lif.lgn.rf_locations import *
import lif.receptive_field.filters.filter_functions as ff

from lif.lgn import cells

from lif.plot import plot
# -
# +
def fig_sq(fig):

    fig = (
        fig
        .update_yaxes(scaleanchor = "x", scaleratio = 1)
        .update_layout(xaxis_constrain='domain', yaxis_constrain='domain')
        )
    return fig
# -

# # Generate RF Locs

# +
st_params = lgn.demo_stparams
lgn_params = lgn.demo_lgnparams

spat_res = st_params.spat_res
# -

# +
spat_filts, temp_filts = cells.mk_filters(lgn_params.n_cells, lgn_params.filters)
# -
# +
rf_distance_scale = rf_locs.mk_rf_locations_distance_scale(
    spat_filters=spat_filts, spat_res=spat_res,
    # This uses the mean ... more correlated to actual SFs and their sizes
    # should smooth out irregular scaling of distance
    use_median_for_pairwise_avg=False,
    magnitude_ratio_for_diameter=None  # relying on default value in settings
    )
rf_locations = cells.mk_rf_locations(
    n=lgn_params.n_cells, rf_loc_params=lgn_params.spread,
    distance_scale=rf_distance_scale
    )
# -
# +
x_coords, y_coords = rf_locations.array_of_coords()
fig_sq(
    px
    .scatter(x=x_coords,y=y_coords )
    ).show()
# -
# +
rf_locs.plot_rf_locations(rf_locations.locations, rf_distance_scale).show()
# -

# # Calculate actual RF Pairwise Distribution

# Cache the 20% coord for each spatial filt
# +
magnitude_ratio_for_diameter = settings.simulation_params.magnitude_ratio_for_rf_loc_scaling
coords_for_target_magnitude = {
        sf.key: spat_filt_coord_at_magnitude_ratio(
            spat_filt=sf.parameters, target_ratio=magnitude_ratio_for_diameter,
            spat_res=spat_res, round=True)
        for key, sf in filters.spatial_filters.items()
    }
# -

# make a layer
# +
lgn_layer = cells.mk_lgn_layer(lgn_params=lgn_params, spat_res=spat_res)
# -

# absolute pairwise distances
# +
x_locs = np.array([c.location.x.mnt for c in lgn_layer.cells])
y_locs = np.array([c.location.y.mnt for c in lgn_layer.cells])
abs_pairwise_dists = rf_locs.pdist(X=np.vstack((x_locs,y_locs)).T, metric='euclidean')
# -
# +
# px.histogram(abs_pairwise_dists, histnorm='probability').show()
# px.histogram(abs_pairwise_dists/rf_distance_scale.mnt, histnorm='probability').show()
# -


# ## Now scale by the diameter of the largest of the pairs
# +
from itertools import combinations, combinations_with_replacement
# -
# +
all_pairs_cell_idxs = tuple(combinations(range(len(lgn_layer.cells)), r=2))
all_pair_max_diam = np.array([
    max(
        coords_for_target_magnitude[lgn_layer.cells[a].spat_filt.key].mnt,
        coords_for_target_magnitude[lgn_layer.cells[b].spat_filt.key].mnt,
        )
    for a,b in all_pairs_cell_idxs
    ])
# -
# +
# px.histogram(abs_pairwise_dists/rf_distance_scale.mnt, histnorm='probability', nbins=20).show()
# px.histogram(abs_pairwise_dists/all_pair_max_diam, histnorm='probability', nbins=50).show()
# -
# +
data_bins = jin_data.distance_vals_insert_lower('on_raw')
jin_data.dist_vals_on_raw
cnts, bins = np.histogram(abs_pairwise_dists/all_pair_max_diam, bins=data_bins)
fig=(
    go.Figure()
    .add_scatter(x=bins[1:], y=cnts/cnts.max(), mode='markers+lines', name='lgn layer')
    .add_scatter(
        x=jin_data.dist_vals_on_raw[:,0],
        y=jin_data.dist_vals_on_raw[:,1]/jin_data.dist_vals_on_raw[:,1].max(),
        mode='markers+lines', name='data')
    )
fig.show()
# -

# ## Multiple LGN Layers to collect population distribution

import math

# +
n_layers = 50
n_combs = math.comb(lgn_params.n_cells, 2)
# all_actual_pairwise_dists = np.empty(shape=n_layers*n_combs)
all_jin_etal_pairwise_dists = np.empty(shape=n_layers*n_combs)

for n in range(n_layers):
    print(n, end='\r')

    lgn_layer = cells.mk_lgn_layer(lgn_params=lgn_params, spat_res=spat_res)

    x_locs = np.array([c.location.x.mnt for c in lgn_layer.cells])
    y_locs = np.array([c.location.y.mnt for c in lgn_layer.cells])

    actual_pairwise_dists = rf_locs.pdist(X=np.vstack((x_locs,y_locs)).T, metric='euclidean')

    all_pairs_cell_idxs = tuple(combinations(range(len(lgn_layer.cells)), r=2))
    all_pair_max_diam = np.array([
        max(
            coords_for_target_magnitude[lgn_layer.cells[a].spat_filt.key].mnt,
            coords_for_target_magnitude[lgn_layer.cells[b].spat_filt.key].mnt,
            )
        for a,b in all_pairs_cell_idxs
        ])

    # all_actual_pairwise_dists[n*n_combs: (n*n_combs)+n_combs] = (actual_pairwise_dists)
    all_jin_etal_pairwise_dists[n*n_combs: (n*n_combs)+n_combs] = (
        actual_pairwise_dists/all_pair_max_diam)

    data_bins = jin_data.distance_vals_insert_lower('on_raw')
# -

# Graph
# +
data_bin_vals = jin_data.dist_vals_on_raw[:,0]
data_prob_vals = jin_data.dist_vals_on_raw[:,1]

jin_etal_cnts, jin_etal_bins = np.histogram(all_jin_etal_pairwise_dists, bins=data_bins)
fig=(
    go.Figure()
    .add_scatter(
        x=data_bin_vals, y=data_prob_vals/data_prob_vals.max(), mode='markers+lines', name='data')
    .add_scatter(
        x=jin_etal_bins[1:], y=jin_etal_cnts/jin_etal_cnts.max(), mode='markers+lines', name='Jin etal empirical dists')
    )
fig.show()
# -

# ## Fit an equation to rescale rf distance

# +
from scipy.optimize import least_squares, OptimizeResult, minimize, minimize_scalar
# -
# +
def custom_rf_dist_scale(
        a: float, b: float, c: float,
        sfs: Tuple[do.DOGSpatialFilter,...], spat_res: ArcLength[scalar]
        ) -> ArcLength[scalar]:

    sf_sd_vals = [sf.parameters.cent.arguments.h_sd.mnt for sf in sfs]

    val = (a * np.mean(sf_sd_vals)) + (b * np.var(sf_sd_vals)) + c

    return ArcLength(val, 'mnt')
# -
# +
def rf_dist_rescaling(x, return_hist_data: bool = False):

    a, b, c = x[0], x[1], x[2]

    n_layers = 1
    # n_layers = 50
    n_combs = math.comb(lgn_params.n_cells, 2)
    # all_actual_pairwise_dists = np.empty(shape=n_layers*n_combs)
    all_jin_etal_pairwise_dists = np.empty(shape=n_layers*n_combs)

    for n in range(n_layers):

        lgn_layer = cells.mk_lgn_layer(
            lgn_params=lgn_params, spat_res=spat_res,
            rf_dist_scale_func=lambda sf, sr: custom_rf_dist_scale(a, b, c, sf, sr))

        x_locs = np.array([c.location.x.mnt for c in lgn_layer.cells])
        y_locs = np.array([c.location.y.mnt for c in lgn_layer.cells])

        actual_pairwise_dists = rf_locs.pdist(X=np.vstack((x_locs,y_locs)).T, metric='euclidean')

        all_pairs_cell_idxs = tuple(combinations(range(len(lgn_layer.cells)), r=2))
        all_pair_max_diam = np.array([
            max(
                coords_for_target_magnitude[lgn_layer.cells[a].spat_filt.key].mnt,
                coords_for_target_magnitude[lgn_layer.cells[b].spat_filt.key].mnt,
                )
            for a,b in all_pairs_cell_idxs
            ])

        # all_actual_pairwise_dists[n*n_combs: (n*n_combs)+n_combs] = (actual_pairwise_dists)
        all_jin_etal_pairwise_dists[n*n_combs: (n*n_combs)+n_combs] = (
            actual_pairwise_dists/all_pair_max_diam)

    data_bins = jin_data.distance_vals_insert_lower('on_raw')
    data_prob_vals = jin_data.dist_vals_on_raw[:,1]

    sum(jin_data.dist_vals_on_raw[:,1] * 0.1)

    jin_etal_cnts, jin_etal_bins = np.histogram(all_jin_etal_pairwise_dists, bins=data_bins)

    if return_hist_data:
        return jin_etal_cnts
    else:
        return sum(np.abs((jin_etal_cnts/jin_etal_cnts.sum()) - data_prob_vals))
# -
# +
opt_res = minimize(rf_dist_rescaling, x0=[1.6, 0, 0], method='Nelder-Mead')
# -
# +
opt_res
# -
math.comb(150, 2)
lgn_params.n_cells = 150

# +
cnts = rf_dist_rescaling(x=[1.6, 0, 0], return_hist_data = False)
# -
# +
data_bin_vals = jin_data.dist_vals_on_raw[:,0]
data_prob_vals = jin_data.dist_vals_on_raw[:,1]
fig=(
    go.Figure()
    .add_scatter(
        x=data_bin_vals, y=data_prob_vals/data_prob_vals.sum(), mode='markers+lines', name='data')
    .add_scatter(
        x=data_bin_vals, y=cnts/cnts.sum(), mode='markers+lines', name='Jin etal empirical dists')
    )
fig.show()
# -

# ## Difficulties ... make the function faster and just rescale each RF by size.

# And ..

# +
lgn_params.n_cells = 20
# -
# +
import random
spat_filts = tuple(filters.spatial_filters.keys())
# -
# +
def rf_dist_rescaling_faster(
        x,
        n_cells,
        spat_res,
        spat_filts,
        location_params: do.LGNLocationParams,
        return_hist_data: bool = False,
        return_all_dists: bool = False):

    # a, b, c = x[0], x[1], x[2]
    a=x

    sf_keys = random.choices(spat_filts, k=n_cells)
    sfs = tuple(filters.spatial_filters[sfk] for sfk in sf_keys)

    # rf_distance_scale = cells.rflocs.mk_rf_locations_distance_scale(
    #         spat_filters=sfs, spat_res=spat_res,
    #         # This uses the mean ... more correlated to actual SFs and their sizes
    #         # should smooth out irregular scaling of distance
    #         use_median_for_pairwise_avg=False,
    #         magnitude_ratio_for_diameter=None  # relying on default value in settings
    #         )

    # rf_distance_scale = a * np.mean([sf.parameters.cent.arguments.h_sd.mnt for sf in sfs])
    # rf_distance_scale = np.array([a*sf.parameters.cent.arguments.h_sd.mnt for sf in sfs])
    rf_distance_scale = np.array([
        a*coords_for_target_magnitude[sf.key].mnt
        for sf in sfs
        ])

    rf_loc_gen = cells.rf_dists.get(location_params.distribution_alias)
    if not rf_loc_gen:
        raise exc.LGNError(f'bad rf loc dist alias')

    x_locs, y_locs = cells.rflocs.mk_unitless_rf_locations(
                        n=n_cells, rf_loc_gen=rf_loc_gen,
                        ratio = location_params.ratio
                        )
    x_locs, y_locs = x_locs * rf_distance_scale, y_locs * rf_distance_scale

    actual_pairwise_dists = rf_locs.pdist(X=np.vstack((x_locs,y_locs)).T, metric='euclidean')

    all_pairs_cell_idxs = combinations(range(n_cells), r=2)
    # all_pairs_cell_idxs = tuple(combinations(range(n_cells), r=2))
    all_pair_max_diam = np.array([
        max(
            coords_for_target_magnitude[sfs[a].key].mnt,
            coords_for_target_magnitude[sfs[b].key].mnt,
            )
        for a,b in all_pairs_cell_idxs
        ])

    # all_actual_pairwise_dists[n*n_combs: (n*n_combs)+n_combs] = (actual_pairwise_dists)
    all_jin_etal_pairwise_dists = actual_pairwise_dists/all_pair_max_diam

    data_bins = jin_data.distance_vals_insert_lower('on_raw')
    data_prob_vals = jin_data.dist_vals_on_raw[:,1]

    jin_etal_cnts, jin_etal_bins = np.histogram(all_jin_etal_pairwise_dists, bins=data_bins)

    if return_all_dists:
        return all_jin_etal_pairwise_dists
    elif return_hist_data:
        return jin_etal_cnts
    else:
        return sum(np.abs((jin_etal_cnts/jin_etal_cnts.sum()) - data_prob_vals))
# -

# ### Trying to handle noise in the objective function ... fit only scalar ?

# +
rf_dist_rescaling_faster(2,
    n_cells=250, spat_res=spat_res, spat_filts=tuple(filters.spatial_filters.keys()),
    location_params=lgn_params.spread)
# -
# +
a_vals = np.arange(1.75, 1.9, 0.02)
errors = [
rf_dist_rescaling_faster([i,0,0],
    n_cells=450, spat_res=spat_res, spat_filts=tuple(filters.spatial_filters.keys()),
    location_params=lgn_params.spread)
    for i in a_vals
]
# -
# +
px.line(x=a_vals, y=errors).show()
# -
math.comb(250,2)
# +
cnts = rf_dist_rescaling_faster(1.8059,
    n_cells=850, spat_res=spat_res, spat_filts=tuple(filters.spatial_filters.keys()),
    location_params=lgn_params.spread, return_hist_data=True)
# -
# +
cnts = rf_dist_rescaling_faster(1.8059,
    n_cells=850, spat_res=spat_res, spat_filts=tuple(filters.spatial_filters.keys()),
    location_params=lgn_params.spread, return_hist_data=True)
# -
lgn_params.spread.ratio=6
# +
all_pw_dists = rf_dist_rescaling_faster(1.8059,
    n_cells=850, spat_res=spat_res, spat_filts=tuple(filters.spatial_filters.keys()),
    location_params=lgn_params.spread, return_all_dists=True)
# -
# +
rf_dist_rescaling_faster([1.79,0,0],
    n_cells=850, spat_res=spat_res, spat_filts=tuple(filters.spatial_filters.keys()),
    location_params=lgn_params.spread, return_hist_data=False)
# -
# +
data_bin_vals = jin_data.dist_vals_on_raw[:,0]
data_prob_vals = jin_data.dist_vals_on_raw[:,1]
fig=(
    go.Figure()
    .add_scatter(
        x=data_bin_vals, y=data_prob_vals/data_prob_vals.sum(), mode='markers+lines', name='data')
    .add_scatter(
        x=data_bin_vals, y=cnts/cnts.sum(), mode='markers+lines', name='Jin etal empirical dists')
    )
fig.show()
# -
# +
data_bin_vals = jin_data.dist_vals_on_raw[:,0]
data_prob_vals = jin_data.dist_vals_on_raw[:,1]

cnts, bins = np.histogram(all_pw_dists, bins=20)

fig=(
    go.Figure()
    .add_scatter(
        x=data_bin_vals, y=data_prob_vals/data_prob_vals.max(), mode='markers+lines', name='data')
    .add_scatter(
        x=bins, y=cnts/cnts.max(), mode='markers+lines', name='Jin etal empirical dists')
    )
fig.show()
# -


# ### Now with scaling only by the magnitude scalar coord

# Try different ratios sets of spat filts

# +
lgn_params.spread.ratio = 2
# -
# +
opt_res = minimize_scalar(
    rf_dist_rescaling_faster, method='Brent',
    args=(350, spat_res, tuple(filters.spatial_filters.keys()), lgn_params.spread)
    )
# -
# +
opt_res
# -
# +
all_pw_dists = rf_dist_rescaling_faster(1.24,
    n_cells=850, spat_res=spat_res, spat_filts=tuple(filters.spatial_filters.keys()),
    location_params=lgn_params.spread, return_all_dists=True)
# -
# +
data_bin_bounds = jin_data.distance_vals_insert_lower('on_raw')
data_bin_vals = jin_data.dist_vals_on_raw[:,0]
data_prob_vals = jin_data.dist_vals_on_raw[:,1]
# data_prob_vals = data_prob_vals/data_prob_vals.max()

# cnts, bins = np.histogram(all_pw_dists, bins=20)
cnts, bins = np.histogram(all_pw_dists, bins=data_bin_bounds)
jin_prob_vals = cnts/cnts.sum()

fig=(
    go.Figure()
    .add_scatter(
        x=data_bin_vals, y=data_prob_vals, mode='markers+lines', name='data')
    .add_scatter(
        x=bins[1:], y=cnts/cnts.sum(), mode='markers+lines', name='Jin etal empirical dists')
    )
fig.show()
# -

# +
lgn_params.spread.ratio = 6
# -
# +
opt_res = minimize_scalar(
    rf_dist_rescaling_faster, method='Brent',
    args=(350, spat_res, tuple(filters.spatial_filters.keys()), lgn_params.spread)
    )
# -
# +
opt_res
# -
# +
all_pw_dists = rf_dist_rescaling_faster(opt_res.x,
    n_cells=850, spat_res=spat_res, spat_filts=tuple(filters.spatial_filters.keys()),
    location_params=lgn_params.spread, return_all_dists=True)
# -
# +
data_bin_bounds = jin_data.distance_vals_insert_lower('on_raw')
data_bin_vals = jin_data.dist_vals_on_raw[:,0]
data_prob_vals = jin_data.dist_vals_on_raw[:,1]
# data_prob_vals = data_prob_vals/data_prob_vals.max()

# cnts, bins = np.histogram(all_pw_dists, bins=data_bin_bounds)
cnts, bins = np.histogram(all_pw_dists, bins=20)
jin_prob_vals = cnts/cnts.sum()

fig=(
    go.Figure()
    .add_scatter(
        x=data_bin_vals, y=data_prob_vals, mode='markers+lines', name='data')
    .add_scatter(
        x=bins[1:], y=cnts/cnts.sum(), mode='markers+lines', name='Jin etal empirical dists')
    )
fig.show()
# -

# trying now with a limited subset of all available spatial filters

# +
subset_spat_filts = ['berardi84_5a', 'berardi84_5b', 'berardi84_6', 'maffei73_2mid',
    'maffei73_2right', 'so81_2bottom', 'so81_5', 'soodak87_1'
    ]
# -
# +
lgn_params.spread.ratio = 2
# -
# +
opt_res = minimize_scalar(
    rf_dist_rescaling_faster, method='Brent',
    args=(
        350, spat_res,
        subset_spat_filts,
        # tuple(filters.spatial_filters.keys()),
        lgn_params.spread
        )
    )
# -
# +
opt_res
# -
# +
all_pw_dists = rf_dist_rescaling_faster(opt_res.x,
    n_cells=850, spat_res=spat_res, spat_filts=tuple(filters.spatial_filters.keys()),
    location_params=lgn_params.spread, return_all_dists=True)
# -
# +
data_bin_bounds = jin_data.distance_vals_insert_lower('on_raw')
data_bin_vals = jin_data.dist_vals_on_raw[:,0]
data_prob_vals = jin_data.dist_vals_on_raw[:,1]
# data_prob_vals = data_prob_vals/data_prob_vals.max()

cnts, bins = np.histogram(all_pw_dists, bins=20)
# cnts, bins = np.histogram(all_pw_dists, bins=data_bin_bounds)
jin_prob_vals = cnts/cnts.sum()

fig=(
    go.Figure()
    .add_scatter(
        x=data_bin_vals, y=data_prob_vals, mode='markers+lines', name='data')
    .add_scatter(
        x=bins[1:], y=cnts/cnts.sum(), mode='markers+lines', name='Jin etal empirical dists')
    )
fig.show()
# -


# Try a 2D fit to include the variance of the spat filt sizes

# +
def rf_dist_rescaling_faster(
        x,
        n_cells,
        spat_res,
        spat_filts,
        location_params: do.LGNLocationParams,
        return_hist_data: bool = False,
        return_all_dists: bool = False):

    # a, b, c = x[0], x[1], x[2]
    # equation: `a + (b*var)`
    a, b = x[0], x[1]

    sf_keys = random.choices(spat_filts, k=n_cells)
    sfs = tuple(filters.spatial_filters[sfk] for sfk in sf_keys)

    # rf_distance_scale = cells.rflocs.mk_rf_locations_distance_scale(
    #         spat_filters=sfs, spat_res=spat_res,
    #         # This uses the mean ... more correlated to actual SFs and their sizes
    #         # should smooth out irregular scaling of distance
    #         use_median_for_pairwise_avg=False,
    #         magnitude_ratio_for_diameter=None  # relying on default value in settings
    #         )

    # rf_distance_scale = a * np.mean([sf.parameters.cent.arguments.h_sd.mnt for sf in sfs])
    # rf_distance_scale = np.array([a*sf.parameters.cent.arguments.h_sd.mnt for sf in sfs])
    all_rf_sizes = np.array([coords_for_target_magnitude[sf.key].mnt for sf in sfs ])
    all_rf_sizes_mean, all_rf_sizes_std = all_rf_sizes.mean(), all_rf_sizes.std()

    # simple scaling static for all RFs
    rf_distance_scale = np.array([
        (a + b*all_rf_sizes_std) * coords_for_target_magnitude[sf.key].mnt
        for sf in sfs
        ])

    # each scaling is specific to the Z score of the RF's size
    # rf_distance_scale = np.array([
    #     ((v:=coords_for_target_magnitude[sf.key].mnt) * a) + (b * (v-all_rf_sizes_mean)/all_rf_sizes_std)
    #     for sf in sfs
    #     ])

    rf_loc_gen = cells.rf_dists.get(location_params.distribution_alias)
    if not rf_loc_gen:
        raise exc.LGNError(f'bad rf loc dist alias')

    x_locs, y_locs = cells.rflocs.mk_unitless_rf_locations(
                        n=n_cells, rf_loc_gen=rf_loc_gen,
                        ratio = location_params.ratio
                        )
    x_locs, y_locs = x_locs * rf_distance_scale, y_locs * rf_distance_scale

    actual_pairwise_dists = rf_locs.pdist(X=np.vstack((x_locs,y_locs)).T, metric='euclidean')

    all_pairs_cell_idxs = combinations(range(n_cells), r=2)
    # all_pairs_cell_idxs = tuple(combinations(range(n_cells), r=2))
    all_pair_max_diam = np.array([
        max(
            coords_for_target_magnitude[sfs[a].key].mnt,
            coords_for_target_magnitude[sfs[b].key].mnt,
            )
        for a,b in all_pairs_cell_idxs
        ])

    # all_actual_pairwise_dists[n*n_combs: (n*n_combs)+n_combs] = (actual_pairwise_dists)
    all_jin_etal_pairwise_dists = actual_pairwise_dists/all_pair_max_diam

    data_bins = jin_data.distance_vals_insert_lower('on_raw')
    data_prob_vals = jin_data.dist_vals_on_raw[:,1]

    jin_etal_cnts, jin_etal_bins = np.histogram(all_jin_etal_pairwise_dists, bins=data_bins)

    if return_all_dists:
        return all_jin_etal_pairwise_dists
    elif return_hist_data:
        return jin_etal_cnts
    else:
        return sum(np.abs((jin_etal_cnts/jin_etal_cnts.sum()) - data_prob_vals))
# -

# +
lgn_params.spread.ratio = 2
# -
# +
opt_res = minimize(
    rf_dist_rescaling_faster, x0=[1, 0], method='Nelder-Mead',
    args=(350, spat_res, tuple(filters.spatial_filters.keys()), lgn_params.spread)
    )
# -
# +
opt_res = minimize_scalar(
    rf_dist_rescaling_faster, method='Brent',
    args=(350, spat_res, tuple(filters.spatial_filters.keys()), lgn_params.spread)
    )
# -
# +
opt_res
# -
# +
subset_spat_filts = ['berardi84_5a', 'berardi84_5b', 'berardi84_6', 'maffei73_2mid',
    'maffei73_2right', 'so81_2bottom', 'so81_5', 'soodak87_1'
    ]
# -
# +
all_pw_dists = rf_dist_rescaling_faster(
    [1.23, 0],
    n_cells=850, spat_res=spat_res,
    spat_filts=subset_spat_filts,
    # spat_filts=tuple(filters.spatial_filters.keys()),
    location_params=lgn_params.spread, return_all_dists=True
    )
# -
# +
data_bin_bounds = jin_data.distance_vals_insert_lower('on_raw')
data_bin_vals = jin_data.dist_vals_on_raw[:,0]
data_prob_vals = jin_data.dist_vals_on_raw[:,1]
# data_prob_vals = data_prob_vals/data_prob_vals.max()

cnts, bins = np.histogram(all_pw_dists, bins=data_bin_bounds)
# cnts, bins = np.histogram(all_pw_dists, bins=20)
jin_prob_vals = cnts/cnts.sum()

fig=(
    go.Figure()
    .add_scatter(
        x=data_bin_vals, y=data_prob_vals, mode='markers+lines', name='data')
    .add_scatter(
        x=bins[1:], y=cnts/cnts.sum(), mode='markers+lines', name='Jin etal empirical dists')
    )
fig.show()
# -


# ### Fitting Rf Dist coefficient Values

# +
def rf_dist_rescaling_faster(
        x,
        n_cells,
        spat_res,
        spat_filts,
        location_params: do.LGNLocationParams,
        return_hist_data: bool = False,
        return_all_dists: bool = False):

    a=x

    sf_keys = random.choices(spat_filts, k=n_cells)
    sfs = tuple(filters.spatial_filters[sfk] for sfk in sf_keys)


    rf_distance_scale = np.array([
        a*coords_for_target_magnitude[sf.key].mnt
        for sf in sfs
        ])

    rf_loc_gen = cells.rf_dists.get(location_params.distribution_alias)
    if not rf_loc_gen:
        raise exc.LGNError(f'bad rf loc dist alias')

    x_locs, y_locs = cells.rflocs.mk_unitless_rf_locations(
                        n=n_cells, rf_loc_gen=rf_loc_gen,
                        ratio = location_params.ratio
                        )
    x_locs, y_locs = x_locs * rf_distance_scale, y_locs * rf_distance_scale

    actual_pairwise_dists = rf_locs.pdist(X=np.vstack((x_locs,y_locs)).T, metric='euclidean')

    all_pairs_cell_idxs = combinations(range(n_cells), r=2)
    # all_pairs_cell_idxs = tuple(combinations(range(n_cells), r=2))
    all_pair_max_diam = np.array([
        max(
            coords_for_target_magnitude[sfs[a].key].mnt,
            coords_for_target_magnitude[sfs[b].key].mnt,
            )
        for a,b in all_pairs_cell_idxs
        ])

    # all_actual_pairwise_dists[n*n_combs: (n*n_combs)+n_combs] = (actual_pairwise_dists)
    all_jin_etal_pairwise_dists = actual_pairwise_dists/all_pair_max_diam

    data_bins = jin_data.distance_vals_insert_lower('on_raw')
    data_prob_vals = jin_data.dist_vals_on_raw[:,1]

    jin_etal_cnts, jin_etal_bins = np.histogram(all_jin_etal_pairwise_dists, bins=data_bins)

    if return_all_dists:
        return all_jin_etal_pairwise_dists
    elif return_hist_data:
        return jin_etal_cnts
    else:
        return sum(np.abs((jin_etal_cnts/jin_etal_cnts.sum()) - data_prob_vals))
# -

# Settling on this subset of RFs ... has a normalish spread of sizes/spat_freq tuning curves
# +
subset_spat_filts = ['berardi84_5a', 'berardi84_5b', 'berardi84_6', 'maffei73_2mid',
    'maffei73_2right', 'so81_2bottom', 'so81_5', 'soodak87_1'
    ]
# -
# +

# -



# ### Testing New Code for fitting all coefficients into lookup vals


# +
subset_spat_filts = [
    'soodak87_1',
    'maffei73_2right',
    'so81_5',
    'berardi84_5a',
    'so81_2bottom',
    'berardi84_5b',
    'maffei73_2mid',
    'berardi84_6',
    ]
subset_spat_filts = ['soodak87_1', 'maffei73_2right', 'so81_5', 'berardi84_5a', 'so81_2bottom', 'berardi84_5b', 'maffei73_2mid', 'berardi84_6']
subset_full_keys = [filters.spatial_filters[k].key for k in subset_spat_filts ]
# -
# +
# data_bins = jin_data.distance_vals_insert_lower('on_raw')
data_bins = jin_data.distance_vals_insert_lower('on_raw', value=0.2)
data_prob_vals = jin_data.dist_vals_on_raw[:,1]
rf_locs.rf_dist_rescaling_residuals(
    2.50, 350, subset_full_keys,
    data_bins, data_prob_vals,
    coords_for_target_magnitude,
    lgn_params.spread
    )
# -
# +
all_dist_dat = rf_locs.rf_dist_rescaling_residuals(
    2.10, 350, subset_full_keys,
    data_bins, data_prob_vals,
    coords_for_target_magnitude,
    lgn_params.spread,
    return_all_dists=True
    )
# -
# +
cnts, bins = np.histogram(all_dist_dat, data_bins)
fig = (
    go.Figure()
    .add_scatter(mode='lines', name='pw_dists_coeff',
        x=bins[1:],
        y=cnts/cnts.sum()
        )
    .add_scatter(mode='lines', name='jin_data',
        x=bins[1:],
        y=data_prob_vals,
        )
    )
fig.show()
# -
# +
[filters.spatial_filters[k].key for k in subset_spat_filts ]
# -
# +
test = rf_locs.mk_rf_loc_scaling_coefficient_lookup_tables(
        subset_full_keys,
        np.array([4]), data_bins, data_prob_vals, lgn_params.spread,
        spat_res, do.RFLocMetaData('jin_et_al', 'on_raw')
    )
# -
# +
test.check_spat_filt_match([filters.spatial_filters[k].key for k in subset_spat_filts])
# -
test.lookup_vals

# !! Just double check that the optimised coefficient is actually a good value!

# +
# a = test.ratio2coefficient(1)
a = 2.047
all_pw_dists = rf_locs.rf_dist_rescaling_residuals(
        a, 850,
        list(test.spat_filt_keys),
        data_bins, data_prob_vals,coords_for_target_magnitude, lgn_params.spread,
        return_all_dists=True
    )

# resid = all_pw_dists = rf_locs.rf_dist_rescaling_residuals(
#         a, 850,
#         list(test.spat_filt_keys),
#         data_bins, data_prob_vals,coords_for_target_magnitude, lgn_params.spread,
#     )
# -
# +
# cnts, bins = np.histogram(all_pw_dists, bins=20)
cnts, bins = np.histogram(all_pw_dists, bins=data_bins)
# -
# +
px.line(x=bins[1:], y=cnts/cnts.sum()).add_scatter(mode='lines', x=data_bins[1:], y=data_prob_vals).show()
# -
# +

# -


# ## Trying different binning of the Jin et al data

# Maybe take Alonso on his suggestion that 0-0.2 and 0.2-0.4 can be separate bins

# +
orig_norm_freq_vals = (
    rf_locs.jin_data.dist_vals_on_raw[:,1] / rf_locs.jin_data.dist_vals_on_raw[:,1].max()
    )

orig_norm_freq_vals_with_additional_bin = np.r_[1, orig_norm_freq_vals]

new_raw_on_prob = (
    orig_norm_freq_vals_with_additional_bin / orig_norm_freq_vals_with_additional_bin.sum()
    )

new_raw_on_bins = np.r_[0, 0.2, rf_locs.jin_data.dist_vals_on_raw[:,0]]

data_bins = jin_data.distance_vals_insert_lower('on_raw')
data_prob = jin_data.dist_vals_on_raw[:,1]

data_bins_min_02 = np.r_[0.2, data_bins[1:]]
# -
# +
px.line(new_raw_on_prob).show()
# -
# +
for ratio in range(1, 10):
    new_res = opt.least_squares(
            rf_locs.bivariate_gauss_pairwise_distance_probability_residuals, x0=[1],
            bounds=([0], [np.inf]),
            args = (ratio, new_raw_on_bins, new_raw_on_prob))
    orig_res = opt.least_squares(
            rf_locs.bivariate_gauss_pairwise_distance_probability_residuals, x0=[1],
            bounds=([0], [np.inf]),
            args = (ratio, data_bins, data_prob))
    print(ratio)
    print(orig_res.cost, orig_res.x[0])
    print(new_res.cost, new_res.x[0])
# -
# +
gauss_params = do.BivariateGaussParams(sigma_x=0.17, ratio=2.)
# -
# +
fig = rf_locs.plot_profile_rf_locations_pairwise_distances(
    gauss_params, new_raw_on_bins, new_raw_on_prob)
fig.show()
# -
# +
gauss_params = do.BivariateGaussParams(sigma_x=0.081, ratio=5.)
# -
# +
fig = rf_locs.plot_profile_rf_locations_pairwise_distances(
    gauss_params, new_raw_on_bins, new_raw_on_prob)
fig.show()
# -
# +
# -

# +
for ratio in range(1, 10):
    new_res = opt.least_squares(
            rf_locs.bivariate_gauss_pairwise_distance_probability_residuals, x0=[1],
            bounds=([0], [np.inf]),
            args = (ratio, data_bins_min_02, data_prob))
    orig_res = opt.least_squares(
            rf_locs.bivariate_gauss_pairwise_distance_probability_residuals, x0=[1],
            bounds=([0], [np.inf]),
            args = (ratio, data_bins, data_prob))
    print(ratio)
    print(orig_res.cost, orig_res.x[0])
    print(new_res.cost, new_res.x[0])
# -
# +
gauss_params = do.BivariateGaussParams(sigma_x=0.185, ratio=2.)
# -
# +
fig = rf_locs.plot_profile_rf_locations_pairwise_distances(
    gauss_params, data_bins_min_02, data_prob)
fig.show()
# -
# +
gauss_params = do.BivariateGaussParams(sigma_x=0.2234, ratio=2.)
# -
# +
fig = rf_locs.plot_profile_rf_locations_pairwise_distances(
    gauss_params, data_bins, data_prob)
fig.show()
# -


# Comparing different new loc generators (0-0.4 and 0.2-0.4 first bin)
# +
loc_gen_on = cells.rf_dists.get('jin_etal_on_raw')
loc_gen_on_02 = cells.rf_dists.get('jin_etal_on_raw_02_bin')
# -
# +
ratios = np.arange(1, 10, 0.2)
sigma_x_vals = [loc_gen_on.ratio2gauss_params(r).sigma_x for r in ratios]
sigma_x_vals_02 = [loc_gen_on_02.ratio2gauss_params(r).sigma_x for r in ratios]
# -
# +
px.line(x=ratios, y=[sigma_x_vals, sigma_x_vals_02]).show()
# -


# # Use combinations or with_replacement when calculating avg pairwise distance unit?


# +
spat_filts, temp_filts = cells.mk_filters(lgn_params.n_cells, lgn_params.filters)
rf_dist_scale = rf_locs.mk_rf_locations_distance_scale(
    spat_filts, spat_res,
    use_with_replacement=True
    )
print(rf_dist_scale.mnt)
# -
# +
rf_locs.avg_largest_pairwise_value(
    range(10), use_median=True, use_with_replacement=False
    )
# -
# +
n = 1000
dist_scales = []
dist_factors_repl = []
dist_factors_no_repl = []
for i in range(n):
    print(i, end='\r')
    spat_filts, temp_filts = cells.mk_filters(lgn_params.n_cells, lgn_params.filters)
    dist_factors_no_repl.append(
        rf_locs.mk_rf_locations_distance_scale(
            spat_filts, spat_res,
            use_median_for_pairwise_avg=False,
            use_with_replacement=False
            ).mnt
    )
    dist_factors_repl.append(
        rf_locs.mk_rf_locations_distance_scale(
            spat_filts, spat_res,
            use_median_for_pairwise_avg=False,
            use_with_replacement=True
            ).mnt
    )
# -
# +
# bins = rf_locs.jin_data.distance_vals_insert_lower(type='on_raw', value=0.2)
# max((max(dist_factors_repl), max(dist_factors_no_repl)))
# min((min(dist_factors_repl), min(dist_factors_no_repl)))
bins = np.arange(20, 70, 5)
dist_factors_repl_cnts, bins = np.histogram(dist_factors_repl, bins=bins)
dist_factors_no_repl_cnts, bins = np.histogram(dist_factors_no_repl, bins=bins)
# -
# +
fig = (
    go.Figure()
    .add_scatter(
        mode='lines',
        x=bins[1:],
        y=dist_factors_repl_cnts,
        name='with replacement'
        )
    .add_scatter(
        mode='lines',
        x=bins[1:],
        y=dist_factors_no_repl_cnts,
        name='withOUT replacement'
        )
    )
fig.show()
# -

# # Testing Different Scaling Parameters

# +
subset_spat_filts = [
    'berardi84_5a',
    'berardi84_5b',
    'berardi84_6',
    'maffei73_2mid',
    'maffei73_2right',
    'so81_2bottom',
    'so81_5',
    'soodak87_1'
 ]
lgn_params.filters.spat_filters = subset_spat_filts
lgn_params.spread.distribution_alias = 'jin_etal_on_raw_02_bin'
# -
# +
coords_for_target_magnitude = rf_locs.mk_all_coords_at_target_magnitude(spat_res)
# -
# +
n = 200
all_pwds_dist_scale = []
all_pwds_coeff = []

for i in range(n):
    print(i, end='\r')

    lgn = cells.mk_lgn_layer(
        lgn_params, spat_res,
        use_dist_scale=True
        )
    pwds = rf_locs.rf_pairwise_distances(
        lgn, spat_res,
        coords_at_target_magnitude=coords_for_target_magnitude
        )
    all_pwds_dist_scale.extend(pwds)

    lgn = cells.mk_lgn_layer(
        lgn_params, spat_res,
        use_dist_scale=False,
        use_spat_filt_size_coefficient=True
        )
    pwds = rf_locs.rf_pairwise_distances(
        lgn, spat_res,
        coords_at_target_magnitude=coords_for_target_magnitude
        )
    all_pwds_coeff.extend(pwds)
# -

# +
bins = rf_locs.jin_data.distance_vals_insert_lower(type='on_raw', value=0.2)
# bins = rf_locs.jin_data.distance_vals_insert_lower(type='on_raw', value=0.0)
pwds_dist_scale_cnts, bins = np.histogram(all_pwds_dist_scale, bins=bins)
pwds_coeff_cnts, bins = np.histogram(all_pwds_coeff, bins=bins)

pwds_dist_scale_prob = pwds_dist_scale_cnts / pwds_dist_scale_cnts.sum()
pwds_coeff_prob = pwds_coeff_cnts / pwds_coeff_cnts.sum()
# -
# +
fig = (
    go.Figure()
    .add_scatter(
        mode='lines',
        x=bins[1:],
        y=pwds_dist_scale_prob ,
        # y=pwds_dist_scale_prob / pwds_dist_scale_prob.max(),
        name='using distance scaling'
        )
    .add_scatter(
        mode='lines',
        x=bins[1:],
        y=pwds_coeff_prob ,
        # y=pwds_coeff_prob / pwds_dist_scale_prob.max(),
        name='using coefficient'
        )
    .add_scatter(
        mode='lines',
        x=bins[1:],
        y=rf_locs.jin_data.dist_vals_on_raw[:,1],
        # y=rf_locs.jin_data.dist_vals_on_raw[:,1] / rf_locs.jin_data.dist_vals_on_raw[:,1].max(),
        name='jin_etal_on_raw'
        )
    )
fig.show()
# -
# +
px.histogram(x=[all_pwds_dist_scale, all_pwds_coeff], barmode='overlay').show()
# -


# ## Simulating different size scales for each member of a pair.

# When normalising to the biggest of the two size-scales, there is clearly spread
# but it is restrained.
# Generally, compared to the "unity" case, there are more longer distances than shorter, but
# they seem limited to being approx 10-20% greater.


# +
n = 10000
x1 = np.random.uniform(size=n)
x2 = np.random.uniform(size=n)
y1 = np.random.uniform(size=n)
y2 = np.random.uniform(size=n)
# -
# +
d = ((x1-x2)**2 + (y1-y2)**2)**0.5
f1 = 3
f2 = 30
d_f = (((f1*x1)-(f2*x2))**2 + ((f1*y1)-(f2*y2))**2)**0.5
# -
# +
(
    px
    .scatter(x=d, y=d_f/f2)
    .add_scatter(mode='lines', x=[0, mx:=max(d.max(), (d_f/f2).max())], y=[0, mx])
    .update_yaxes(scaleanchor = "x", scaleratio = 1, row=1, col=1)
    ).show()
# -
# +
((d_f/f2)/d).mean()
# -


# +
n = 10000
x1 = np.random.uniform(size=n)
x2 = np.random.uniform(size=n)
y1 = np.random.uniform(size=n)
y2 = np.random.uniform(size=n)
# -
# +
mean_ratios = []
median_ratios = []
for i in range(1, 11):
    print(i, end='\r')
    d = ((x1-x2)**2 + (y1-y2)**2)**0.5
    f1 = 3
    f2 = i * f1
    d_f = (((f1*x1)-(f2*x2))**2 + ((f1*y1)-(f2*y2))**2)**0.5
    mean_ratios.append(((d_f/f2)/d).mean())
    median_ratios.append( np.median((d_f/f2)/d) )
# -
# +
px.scatter(x=range(1,11), y = [mean_ratios, median_ratios]).show()
# -

# When the size differences are randomised, the difference is much more stable.
# The median ratio seems to stay close to a ratio of 1.
# As the potential or maximum difference in RF size increases, the mean escalates but seems
# to plateau at a ratio of about ~ 1.4.

# +
mean_ratios = []
median_ratios = []
high_factor_range = list(range(2, 15))
for i in high_factor_range:
    print(i, end='\r')
    d = ((x1-x2)**2 + (y1-y2)**2)**0.5

    f = np.random.randint(low=1, high=i, size=(2,n))
    f1 = f[0,:]
    f2 = f[1,:]

    d_f = (((f1*x1)-(f2*x2))**2 + ((f1*y1)-(f2*y2))**2)**0.5

    pair_max_f = f.max(axis=0)
    ratios = (d_f/pair_max_f)/d
    mean_ratios.append(ratios.mean())
    median_ratios.append(np.median(ratios))
# -
# +
px.scatter(x=high_factor_range, y = [mean_ratios, median_ratios]).show()
# -
# +
px.scatter(x=d, y=(d_f/pair_max_f)).show()
# -


# Graph
# +
data_bin_vals = jin_data.dist_vals_on_raw[:,0]
data_prob_vals = jin_data.dist_vals_on_raw[:,1]

jin_etal_cnts, jin_etal_bins = np.histogram(all_jin_etal_pairwise_dists, bins=data_bins)
fig=(
    go.Figure()
    .add_scatter(
        x=data_bin_vals, y=data_prob_vals/data_prob_vals.max(), mode='markers+lines', name='data')
    .add_scatter(
        x=jin_etal_bins[1:], y=jin_etal_cnts/jin_etal_cnts.max(), mode='markers+lines', name='Jin etal empirical dists')
    )
fig.show()
# -




# So they're both done in the same order
# +
dists_close = []
for n in range(len(all_pairs_cell_idxs)):
    a,b = all_pairs_cell_idxs[n]
    cell_a, cell_b = lgn_layer.cells[a], lgn_layer.cells[b]
    distance = (
        (cell_a.location.x.mnt-cell_b.location.x.mnt)**2 +
        (cell_a.location.y.mnt-cell_b.location.y.mnt)**2
        )**0.5
    dists_close.append(np.isclose(distance, abs_pairwise_dists[n]))
all(dists_close)
# -

x_locs = np.random.normal(size=n_simulated_locs, scale=gauss_params.sigma_x)
y_locs = np.random.normal(size=n_simulated_locs, scale=gauss_params.sigma_y)
emp_pwdists = pdist(X=np.vstack((x_locs, y_locs)).T, metric='euclidean')




# +
rf_locs.plot_jin_data_with_raw_data(rf_locs.jin_data).show()
# -


