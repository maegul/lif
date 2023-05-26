# # Imports

from pathlib import Path
from typing import Sequence, List, Dict, Optional

from dash import Dash, html, dcc, Input, Output, State, ctx, dash_table
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
from lif.simulation import run


# # App!
app = Dash(__name__)

# # Contents

results_dir = Path('/home/ubuntu/lif_hws/work/results_data')

# ## All Experiment Folders

def mk_all_exp_dir_listing():

	all_exp_dir = list(run.mk_all_exp_dir(results_dir))

	all_meta_data = [
		run.load_meta_data(exp_dir)
		for exp_dir in all_exp_dir
	]

	all_extra_info = [
		f"{meta_data.get('comments', 'No cmnt')} ({meta_data.get('creation_time', 'No time')})"
		for meta_data in all_meta_data
	]

	listing = [
		{
			'label': f'{exp_dir.name} - {extra_info}',
			'value': exp_dir.resolve().as_posix()
		}
		for exp_dir, extra_info in zip(all_exp_dir, all_extra_info)
	]

	return listing


# ### Meta data about experiment

# eg, number of simulations, params etc

# ### View Simulation Results

# Select simulation
# select results view from list (will trigger a results view function that will return graphs etc)
# eg:
#  - lgn spikes + v1 spikes
#  - lgn spikes + v1 memb pot
#  -

# # Layout

app.layout = (html.Div(children= [
	html.H1('Results Explorer'),
	dcc.Dropdown(
		options=mk_all_exp_dir_listing(),
		id='exp_dir'
		)


	]
	) )



# # Run
if __name__ == '__main__':
	# app.run_server(debug=True)
	app.run(debug=True)


# # Basic Prototyping results analysis

# +
# run.mk_all_exp_dir = mk_all_exp_dir
# run.load_meta_data = load_meta_data
# -
# +
exp_dirs = list(run.mk_all_exp_dir(results_dir))
# -
# +
listings = mk_all_exp_dir_listing()
# -
# +
meta_data, sim_results = run.load_simulation_results(results_dir, Path(listings[0]['value']))
# -
# +
stim_keys = list(sim_results.results.keys())
len(stim_keys)
# -
# +
stim_key = stim_keys[0]
# -
# +
test = sim_results.results[stim_key][0].get_spikes(0)
# -
# +
temp_ext = sim_results.params.space_time_params.temp_ext
bin_width = Time(20, 'ms')
bins = np.arange(0, temp_ext.ms, bin_width.ms)
# -
# +
total_epsc_v1_cnts = {}
for exp in listings:
	print(exp['label'])

	meta_data, sim_results = run.load_simulation_results(results_dir, Path(exp['value']))
	aggregate_bin_counts = np.zeros_like(bins[:-1])

	for i, sim_result in enumerate(sim_results.results[stim_key]):
		print(i, end='\r')
		for n_trial in range(sim_results.params.n_trials):
			cnts, hist_bins = np.histogram(
					sim_result.get_spikes(n_trial) / run.lif_model.bnun.msecond,
					bins
				)
			print(cnts)
			aggregate_bin_counts += cnts

	aggregate_bin_counts /= (sim_results.params.n_simulations * sim_results.params.n_trials)
	aggregate_bin_counts /= bin_width.s

	total_epsc_v1_cnts[sim_results.params.lif_params.total_EPSC] = aggregate_bin_counts
# -
# +
total_epsc_v1_cnts
# -
# +
run._save_pickle_file(results_dir/'total_epsc_v1_avg_rate.pkl', (bins, total_epsc_v1_cnts))
# -
ll
