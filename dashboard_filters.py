"""Dashboard for viewing the characteristics of any of the saved/available
spatial and temporal filters.
"""

from pathlib import Path

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import numpy as np
import pandas as pd

# from lif import plot, ff, DOGSpatialFilter
from lif.plot import plot
from lif.receptive_field.filters import filter_functions as ff, filters
from lif.utils.data_objects import DOGSpatialFilter

app = Dash(__name__)

# # get all saved filters
spatial_filters = filters.spatial_filters
saved_sfs_dict = filters.spatial_filters
# saved_sfs_dict = {key: spatial_filters[key] for key in sorted(spatial_filters)}
# saved_sfs = sorted(DOGSpatialFilter.get_saved_filters(), key=lambda p : p.name)
# saved_sfs_dict = {
#     ssf.name: DOGSpatialFilter.load(ssf)
#     for ssf in saved_sfs
# }

# # dropdown box to select spatial filter
sf_selector = dcc.Dropdown(
    options = [
        {
            'label':f'{ssf.source_data.meta_data.author} ({ssf.source_data.meta_data.year}), {ssf.source_data.meta_data.reference}',  # type: ignore
            'value': name
        }
            for name, ssf in saved_sfs_dict.items()
    ],
    # options = list(saved_sfs_dict.keys()),
    value = list(saved_sfs_dict.keys())[0],
    id='sf_sel')

# # Main elements
sf_selected = html.H3(id='sf_selected')

spat_filt_fig = dcc.Graph(id='spat_filt', )
spat_filt_fit_fig = dcc.Graph(id='spat_filt_fit', )
spat_filt_heatmap_fig = dcc.Graph(id='spat_filt_hm')

spat_filt_range_slid = dcc.Slider(10, 1000, value=50,
    tooltip={"placement": "bottom", "always_visible": True},
    id='spat_filt_range_slid')

# # callback that replots for the selected filter
@app.callback(
    Output('sf_selected', 'children'),
    Output('spat_filt', 'figure'),
    Output('spat_filt_fit', 'figure'),
    Output('spat_filt_hm', 'figure'),
    Input('sf_sel', 'value'),
    Input('spat_filt_range_slid', 'value')
)
def display_spat_filt(value: str, sf_range_value):
    # sf = DOGSpatialFilter.load(saved_sfs_dict[value])
    sf = saved_sfs_dict[value]
    sf_key = sf.source_data.meta_data.make_key()  # type: ignore
    spat_filt = plot.spat_filt(sf, sd=True)
    spat_filt = (
        spat_filt
        .update_layout(xaxis_range=(-sf_range_value, sf_range_value))
        )

    spat_filt_fit = plot.spat_filt_fit(sf)
    ori_sf = ff.mk_ori_biased_spatfilt_params_from_spat_filt(sf, circ_var=0.8)
    ori_sf_heatmap = plot.ori_spat_freq_heatmap(
        ori_sf, n_orientations=16,
        log_yaxis_start_exponent=-1)

    return (
        f'SF: {sf_key}',
        spat_filt,
        spat_filt_fit,
        ori_sf_heatmap
        )

all_spat_filts_norm_mag = dcc.RadioItems(
    options = [
        {'label': 'True', 'value':True},
        {'label': 'False', 'value':False},
        ],
        value=False,
    id='all_spat_filts_norm_mag'
    )

all_spat_filts_graph = dcc.Graph(
        figure = plot.multi_spat_filt(
            spat_filters=list(saved_sfs_dict.values()),
            sd=True
            ),
        id='all_spat_filt'
        )


all_spat_filts_ft_norm_mag = dcc.RadioItems(
    options = [
        {'label': 'True', 'value':True},
        {'label': 'False', 'value':False},
        ],
        value=True,
    id='all_spat_filts_ft_norm_mag'
    )
all_spat_filts_ft_freq_share = dcc.RadioItems(
    options = [
        {'label': 'True', 'value':True},
        {'label': 'False', 'value':False},
        ],
        value=True,
    id='all_spat_filts_ft_freq_share'
    )

all_spat_filts_ft_graph = dcc.Graph(
        figure = plot.multi_spat_filt_fit(
            fit_filters=list(saved_sfs_dict.values()),
            normalise_magnitude=True,
            share_freq_bounds=True
            ),
        id='all_spat_filt_ft'
        )


# # Display all spat filts

@app.callback(
    Output('all_spat_filt', 'figure'),
    Input('all_spat_filts_norm_mag', 'value')
    )
def display_all_spat_filts(norm_mag: bool):

    fig = plot.multi_spat_filt(
                spat_filters=list(saved_sfs_dict.values()),
                sd=True,
                normalise_magnitude=norm_mag
                )
    return fig

@app.callback(
    Output('all_spat_filt_ft', 'figure'),
    Input('all_spat_filts_ft_norm_mag', 'value'),
    Input('all_spat_filts_ft_freq_share', 'value')
    )
def display_all_spat_filts_fts(
        norm_mag: bool, share_freqs: bool
        ):

    fig = plot.multi_spat_filt_fit(
            fit_filters=list(saved_sfs_dict.values()),
            normalise_magnitude=norm_mag,
            share_freq_bounds=share_freqs
            )
    return fig

app.layout = html.Div(children=[
    sf_selector,
    sf_selected,
    html.Div(children=[
        html.Div(children='Spat filt spatial range (from 0)'),
        spat_filt_range_slid,
        ]
        ),
    html.Div(
        children = [
            spat_filt_fig,
            spat_filt_fit_fig,
            spat_filt_heatmap_fig
            ],
        style={'display': 'flex'}
            ),

    html.Div(children=[html.Span('Normalise magnitudes to 1?'), all_spat_filts_norm_mag]),
    all_spat_filts_graph,

    html.Div(children=[
        html.Span('Normalise magnitudes to 1?'),
        all_spat_filts_ft_norm_mag,
        html.Span('Share min and max freqs for all filters?'),
        all_spat_filts_ft_freq_share
        ]),
    all_spat_filts_ft_graph
])


if __name__ == '__main__':
    app.run_server(debug=True)
