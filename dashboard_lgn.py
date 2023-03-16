from pathlib import Path
from typing import Sequence, List, Dict, Optional

from dash import Dash, html, dcc, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# from lif import plot, ff, DOGSpatialFilter
from lif.plot import plot
from lif.receptive_field.filters import filter_functions as ff, filters

from lif.lgn import demo_lgnparams, demo_stparams, rf_locations as rf_locs


from lif.lgn import cells
from lif.stimulus import stimulus
from lif.convolution import convolve
from lif.simulation import all_filter_actual_max_f1_amp as all_max_f1

from lif.utils import data_objects as do
from lif.utils.units.units import ArcLength, SpatFrequency, TempFrequency, Time

app = Dash(__name__)

# # get all saved filters
spatial_filters = filters.spatial_filters


# unique_spat_filts = set(filters.spatial_filters.keys())
spat_filt_colors = plot.spat_filt_colors
# spat_filt_colors = {
# 		key: px.colors.qualitative.Dark24[i]
# 		for i, key in enumerate(filters.spatial_filters.keys())
# 	}


# # Dummy LGN
demo_stparams = do.SpaceTimeParams(
    spat_ext=ArcLength(660, 'mnt'), spat_res=ArcLength(1, 'mnt'), 
	temp_ext=Time(1, 's'), temp_res=Time(1, 'ms'), 
	array_dtype='float32')

lgn = cells.mk_lgn_layer(demo_lgnparams, demo_stparams.spat_res, force_central=False)

lgn_rec_fields_fig = dcc.Graph(
	figure = plot.lgn_sf_locations_and_shapes(lgn),
	id='lgn_rec_fields',
	style={'width': '70vh', 'height': '70vh'}
	)


all_coords_at_magnitude = rf_locs.mk_all_coords_at_target_magnitude(
	spat_res=demo_stparams.spat_res)

def make_rec_pwds_fig(lgn, spat_res, all_coords):
	fig = px.violin(
		x= rf_locs.rf_pairwise_distances(
				lgn_layer=lgn, spat_res=spat_res,
				coords_at_target_magnitude=all_coords,
				),
		points='all'
			)
	return fig

lgn_rec_pwds = dcc.Graph(
	figure = make_rec_pwds_fig(lgn, demo_stparams.spat_res, all_coords_at_magnitude),
	id='lgn_rec_field_pwds',
	style={'width': '70vh', 'height': '70vh'}
	)

lgn_response_fig = dcc.Graph(
	id='lgn_response',
	figure=go.Figure(),
	style={'width': '70vh', 'height': '70vh'}
	)


# # Reload a new LGN

lgn_rec_fields_reload = html.Button('New LGN Layer', id='reload_lgn_rec_fields_button', n_clicks=0)

# # List Cells

def make_cell_list_options(lgn: do.LGNLayer):



	# ensure value index represents location in lgn.cells so that it will relate
	# to all other logic relying on the index/position in lgn.cells
	cell_list_opts = [
		{
			'label': filters.reverse_spatial_filters[c.spat_filt.key],
			'value': i,
			'y_val': c.location.y.mnt
		}
		for i, c in enumerate(lgn.cells)
		]

	# now sort by location
	cell_list_opts = [
		{
			'label': html.Span(
				opt['label'], style={'color': spat_filt_colors[opt['label']]}),
			'value': opt['value']
		}
		for opt in sorted(cell_list_opts, key=lambda c: c['y_val'], reverse=True)
	]
	# cell_list_opts = [
	# 	{k: v for k,v in opt.items() if k != 'y_val'}
	# 	for opt in sorted(cell_list_opts, key=lambda c: c['y_val'], reverse=True)
	# ]

	return cell_list_opts

# dummy
lgn_layer_cell_list = dcc.Checklist(
	options = make_cell_list_options(lgn),
	value=[],
	inline=False,
	style={'display': 'flex', 'flex-direction': 'column'},
	id='lgn_layer_cell_list'
	)


# # LGN Params

lgn_params_dial_label_style = {
	'font-family': 'sans-serif',
	'border-bottom': '1px solid grey',
	'margin': '5px',
	'margin-right': '10px'
}

lgn_params_dials = html.Div(children = [
		html.Div(children=[
			html.Span('RF Line coloring', style=lgn_params_dial_label_style),
			dcc.RadioItems(
				options=[
					{'label': 'Colored', 'value': True},
					{'label': 'Black', 'value': False}
				],
				value=False,
				inline=True, id='lgn_params_rf_line_color'),
			]),
		html.Div(children=[
			html.Span('N cells', style=lgn_params_dial_label_style),
			dcc.Input(id='lgn_params_n_cells', type='number', min=1, max=50, step=1, value=20),
			]),
		html.Div(children=[
			html.Span('Mean Orientation', style=lgn_params_dial_label_style),
			dcc.Input(id='lgn_params_ori', type='number', min=0, max=90, step=1, value=0),
			]),
		html.Div(children=[
			html.Span('Circ Var (of oris)', style=lgn_params_dial_label_style),
			dcc.Slider(0, 1, 0.05, value=0.5, id='lgn_params_ori_cv'),
			]),
		html.Div(children=[
			html.Span('Circ Var (of cells) distribution', style=lgn_params_dial_label_style),
			dcc.Dropdown(
				list(do.AllCircVarianceDistributions.__dataclass_fields__.keys()),
				'naito_lg_highsf', id='lgn_params_cv_dist'
				),
			]),
		html.Div(children=[
			html.Span('Circ Var Def method', style=lgn_params_dial_label_style),
			dcc.Dropdown(
				do.CircularVarianceParams.all_methods(),
				'naito', id='lgn_params_cv_def'
				),
			]),
		html.Div(children=[
			html.Span('Loc ratio', style=lgn_params_dial_label_style),
			dcc.Input(id='lgn_params_loc_ratio', type='number', min=1, max=10, step=0.5, value=2),
			]),
		html.Div(children=[
			html.Span('Use Dist Scale or Coeff', style=lgn_params_dial_label_style),
			dcc.RadioItems(
				options=[
					{'label': 'Dist Scale', 'value': True},
					{'label': 'Coeff', 'value': False}
				],
				value=True,
				inline=True, id='lgn_params_dist_scale_or_coeff'),
			]),
		html.Div(children=[
			html.Span('Use RF Coord at Mag', style=lgn_params_dial_label_style),
			dcc.RadioItems(
				options=[
					{'label': 'Yes', 'value': True},
					{'label': 'Use 2SD', 'value': False}
				],
				value=True,
				inline=True, id='lgn_params_rf_use_coord_at_mag'),
			]),
		html.Div(children=[
			html.Span('Loc Distribution', style=lgn_params_dial_label_style),
			dcc.Dropdown(
				list(cells.rf_dists.keys()),
				'jin_etal_on_raw_02_bin', id='lgn_params_loc_dist'
				),
			]),
		html.Div(children=[
			html.Span('Loc Ori', style=lgn_params_dial_label_style),
			dcc.Input(id='lgn_params_loc_ori', type='number', min=0, max=90, step=1, value=90),
			]),
		html.Div(children=[
			html.Span('Spat Filters', style=lgn_params_dial_label_style),
			dcc.Checklist(
				options = ['all'] + list(
					sorted(
						filters.spatial_filters.keys(),
						# sort by size of center
						key=lambda k: filters.spatial_filters[k].parameters.cent.array()[-1]
						)
					),
				value=['all'],
				inline=False,
				style={'display': 'flex', 'flex-direction': 'column'},
				id='lgn_params_spatial_filters'
				)
			]),
	])


# # Stim Params

lgn_stim_dials = html.Div(children = [
		html.Button('Convolve!',
			id='convolve_with_stim', n_clicks=0),
		html.Div(children=[
			html.Span('Spat Freq', style=lgn_params_dial_label_style),
            dcc.Slider(0, 10, 0.1, value=0.8, id='lgn_stim_params_sf', 
                marks=None, tooltip={"placement": "bottom", "always_visible": True}),
		]),
		html.Div(children=[
			html.Span('Temp Freq', style=lgn_params_dial_label_style),
			dcc.Slider(0, 10, 0.1, value=4, id='lgn_stim_params_tf',
                marks=None, tooltip={"placement": "bottom", "always_visible": True}),
		]),
		html.Div(children=[
			html.Span('Orientation', style=lgn_params_dial_label_style),
			dcc.Slider(0, 180, 1, value=90, id='lgn_stim_params_ori',
                marks=None, tooltip={"placement": "bottom", "always_visible": True}),
		]),
		html.Div(children=[
			html.Span('Amp', style=lgn_params_dial_label_style),
			dcc.Input(id='lgn_stim_params_amp', type='number', min=0, max=10, step=0.2, value=1),
		]),
		html.Div(children=[
			html.Span('DC', style=lgn_params_dial_label_style),
			dcc.Input(id='lgn_stim_params_dc', type='number', min=0, max=10, step=0.2, value=1),
		]),
		html.Div(children=[
			html.Span('Contrast', style=lgn_params_dial_label_style),
			dcc.Input(id='lgn_stim_params_cont', type='number', min=0, max=1, step=0.1, value=0.4),
		]),
	])

# # New Figure Callback

@app.callback(
	Output('lgn_rec_fields', 'figure'),
	Output('lgn_layer_cell_list', 'options'),
	Output('lgn_layer_cell_list', 'value'),
	Output('lgn_rec_field_pwds', 'figure'),
	Input('reload_lgn_rec_fields_button', 'n_clicks'),
	Input('lgn_layer_cell_list', 'value'),
	Input('lgn_params_rf_line_color', 'value'),
	# lgn params
	State('lgn_params_n_cells', 'value'),
	State('lgn_params_ori', 'value'),
	State('lgn_params_ori_cv', 'value'),
	State('lgn_params_cv_dist', 'value'),
	State('lgn_params_cv_def', 'value'),
	State('lgn_params_loc_ratio', 'value'),
	State('lgn_params_dist_scale_or_coeff', 'value'),
	State('lgn_params_rf_use_coord_at_mag', 'value'),
	State('lgn_params_loc_dist', 'value'),
	State('lgn_params_loc_ori', 'value'),
	State('lgn_params_spatial_filters', 'value'),
	)
def reload_lgn_rec_fields(
		_,
		selected_cells,
		rf_line_color,
		lgn_params_n_cells,
		lgn_params_ori,
		lgn_params_ori_cv,
		lgn_params_cv_dist,
		lgn_params_cv_def,
		lgn_params_loc_ratio,
		lgn_params_dist_scale_or_coeff,
		lgn_params_use_coord_at_mag,
		lgn_params_loc_dist,
		lgn_params_loc_ori,
		lgn_params_spatial_filters,
		):

	input_trigger_id = ctx.triggered_id
	# print('\n*****\n',
	# 	input_trigger_id,
	# 	selected_cells,
	# 	lgn_params_n_cells,
	# 	lgn_params_ori,
	# 	lgn_params_ori_cv,
	# 	lgn_params_cv_dist,
	# 	lgn_params_cv_def,
	# 	lgn_params_loc_ratio,
	# 	lgn_params_loc_dist,
	# 	lgn_params_loc_ori,
	# 	lgn_params_spatial_filters,
	# 	)

	global lgn
	# reloading LGN not highlighting a cell ... remake lgn
	if input_trigger_id == 'reload_lgn_rec_fields_button':
		lgn_params = do.LGNParams(
				n_cells=lgn_params_n_cells,
				orientation=do.LGNOrientationParams(
					ArcLength(lgn_params_ori, 'deg'), lgn_params_ori_cv),
				circ_var=do.LGNCircVarParams(lgn_params_cv_dist, lgn_params_cv_def),
				spread=do.LGNLocationParams(
					lgn_params_loc_ratio, lgn_params_loc_dist,
					orientation=ArcLength(lgn_params_loc_ori, 'deg')),
				filters=do.LGNFilterParams(
					spat_filters = 'all' if 'all' in lgn_params_spatial_filters else lgn_params_spatial_filters,
					temp_filters='all'),
				F1_amps=do.LGNF1AmpDistParams()
			)
		lgn = cells.mk_lgn_layer(
			lgn_params, demo_stparams.spat_res, force_central=False,
			use_dist_scale = lgn_params_dist_scale_or_coeff,
			use_spat_filt_size_coefficient=True if lgn_params_dist_scale_or_coeff is False else False
			)
		cell_list_value = []
	else:
		cell_list_value = selected_cells

	fig = plot.lgn_sf_locations_and_shapes(
		lgn, highlight_idxs=cell_list_value, color_ellipses=rf_line_color,
		coords_at_magnitude=(
			all_coords_at_magnitude
				if lgn_params_use_coord_at_mag else
				None
			))

	fig_pwds = (
		make_rec_pwds_fig(lgn, demo_stparams.spat_res, all_coords_at_magnitude)
		.update_layout(xaxis_range = (0, 3))
		)

	cell_list_opts = make_cell_list_options(lgn)

	return fig, cell_list_opts, cell_list_value, fig_pwds


# # Rate Curves and Stimulus
@app.callback(
	Output('lgn_response', 'figure'),
	Input('convolve_with_stim', 'n_clicks'),
	State('lgn_stim_params_sf', 'value'),
	State('lgn_stim_params_tf', 'value'),
	State('lgn_stim_params_ori', 'value'),
	State('lgn_stim_params_amp', 'value'),
	State('lgn_stim_params_dc', 'value'),
	State('lgn_stim_params_cont', 'value'),
	)
def make_lgn_stimulus_response(
		n_clicks,
		lgn_stim_params_sf,
		lgn_stim_params_tf,
		lgn_stim_params_ori,
		lgn_stim_params_amp,
		lgn_stim_params_dc,
		lgn_stim_params_cont,
		):

#    input_trigger_id = ctx.triggered_id
#    if input_trigger_id == 'reload_':

	print(f'Convolve clicked: {n_clicks}')

	stim_params = do.GratingStimulusParams(
		SpatFrequency(lgn_stim_params_sf, 'cpd'),
		TempFrequency(lgn_stim_params_tf, 'hz'),
		ArcLength(lgn_stim_params_ori, 'deg'),
		amplitude=lgn_stim_params_amp,
		DC=lgn_stim_params_dc,
		contrast=do.ContrastValue(lgn_stim_params_cont)
		)

	# just make the stimulus if not created already
	stimulus.mk_stimulus_cache(demo_stparams, tuple([stim_params]))

	spat_filts: Sequence[np.ndarray] = []
	temp_filts: Sequence[np.ndarray] = []
	responses: Sequence[do.ConvolutionResponse] = []

	actual_max_f1_amps = all_max_f1.mk_actual_max_f1_amps(stim_params=stim_params)

	stim_array = stimulus.load_stimulus_from_params(demo_stparams, stim_params)

	# ##### Loop through LGN cells
	for cell in lgn.cells:
		# spatial filter array
		xc, yc = ff.mk_spat_coords(
					demo_stparams.spat_res,
					sd=cell.spat_filt.parameters.max_sd()
					)
		spat_filt = ff.mk_dog_sf(x_coords=xc, y_coords=yc, dog_args=cell.spat_filt)
		# Rotate array
		spat_filt = ff.mk_oriented_sf(spat_filt, cell.orientation)

		spat_filts.append(spat_filt)

		# temporal filter array
		tc = ff.mk_temp_coords(
			demo_stparams.temp_res,
			tau=cell.temp_filt.parameters.arguments.tau
			)
		temp_filt = ff.mk_tq_tf(tc, cell.temp_filt)
		temp_filts.append(temp_filt)

		# slice stimulus
		spat_slice_idxs = stimulus.mk_rf_stim_spatial_slice_idxs(
			demo_stparams, cell.spat_filt, cell.location)
		stim_slice = stimulus.mk_stimulus_slice_array(
			demo_stparams, stim_array, spat_slice_idxs)

		# convolve
		actual_max_f1_amp = all_max_f1.get_cell_actual_max_f1_amp(cell, actual_max_f1_amps)
		cell_resp = convolve.mk_single_sf_tf_response(
				demo_stparams, cell.spat_filt, cell.temp_filt,
				spat_filt, temp_filt,
				stim_params, stim_slice,
				filter_actual_max_f1_amp=actual_max_f1_amp.value,
				target_max_f1_amp=cell.max_f1_amplitude
				)

		responses.append(cell_resp)

	# be paranoid and use tuples ... ?
	# spat_filts, temp_filts, responses = (
	# 	tuple(spat_filts), tuple(temp_filts), tuple(responses)
	# 	)

	# #### Poisson spikes for all cells
	# Sigh ... the array is stored along with the adjustment params in an object
	# ... and they're all called "response(s)"
	response_arrays = tuple(
			response.response for response in responses
		)
	# lgn_layer_responses = convolve.mk_lgn_response_spikes(
	# 		demo_stparams, response_arrays
	# 	)

	fig = go.Figure()

	for r in response_arrays:
		fig.add_scatter(
			mode='lines',
			y=r
			)

	return fig


# # Layout

app.layout = (
	html.Div(children = [
		html.Div(children = [
			lgn_rec_fields_fig,
			lgn_rec_fields_reload,
			lgn_params_dials,
            lgn_stim_dials,
            lgn_response_fig
		]),
		lgn_layer_cell_list,
		lgn_rec_pwds,
	],
	style={'display': 'flex'}
	)
)

if __name__ == '__main__':
	# app.run_server(debug=True)
	app.run(debug=True)
