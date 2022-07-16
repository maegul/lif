"""Dashboard for viewing the characteristics of any of the saved/available
spatial and temporal filters.
"""

from pathlib import Path

from dash import Dash, html, dcc
import plotly.express as px
import numpy as np
import pandas as pd

from lif import plot, ff, DOGSpatialFilter

app = Dash(__name__)

saved_sfs = DOGSpatialFilter.get_saved_filters()


sf = DOGSpatialFilter.load(DOGSpatialFilter.get_saved_filters()[0])
spat_filt = plot.spat_filt(sf)
spat_filt_fit = plot.spat_filt_fit(sf)
spat_filt_circ_var_matrix = (
	plot.orientation_circ_var_subplots(sf,
		spat_freq_factors=np.array([0.5, 1, 2, 4]),
		circ_vars=np.array([0.1, 0.3, 0.5, 0.7, 0.8, 0.9]))
	.update_layout(height=800))


ori_sf = ff.mk_ori_biased_spatfilt_params_from_spat_filt(sf, circ_var=0.8)
ori_sf_heatmap = plot.ori_spat_freq_heatmap(ori_sf, n_orientations=16 )

app.layout = html.Div(children=[
	html.H1(children='Spatial Filter Profile'),
	html.Div(children=[html.Div(ssf.name) for ssf in saved_sfs]),
	dcc.Graph(
		id='spat_filt',
		figure=spat_filt
	),
	dcc.Graph(
		id='spat_filt_fit',
		figure=spat_filt_fit
	),
	dcc.Graph(
		id='circ_var_matrix', figure=spat_filt_circ_var_matrix),
	dcc.Graph(id='ori_sf_heatmap', figure=ori_sf_heatmap)
])


if __name__ == '__main__':
	app.run_server(debug=True)
