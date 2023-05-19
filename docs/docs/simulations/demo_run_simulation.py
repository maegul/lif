

# # Imports
# +
from pathlib import Path
import re
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
import lif.stimulus.stimulus as stimulus
import lif.convolution.convolve as convolve
import lif.convolution.correction as correction
import lif.convolution.estimate_real_amp_from_f1 as est_f1
import lif.receptive_field.filters.filter_functions as ff

from lif.lgn import cells

import lif.simulation.all_filter_actual_max_f1_amp as all_max_f1
import lif.simulation.leaky_int_fire as lifv1
from lif.simulation import run

from lif.plot import plot
# -


# # Stimulus generation and managements

# ## Estimating Required Width

# +
spat_res = ArcLength(1, 'mnt')
lgn_params = lgn.demo_lgnparams
max_spat_ext = stimulus.estimate_max_stimulus_spatial_ext_for_lgn(
	spat_res, lgn_params, n_cells=1000, safety_margin_increment=0.1)
print(max_spat_ext.deg)
# -
# +
print(f'Number of pixels in width: {max_spat_ext.base / spat_res.base}')
# -

# ## size of stimulus this would require
# +
size_factor = 1.1
# spat_ext=ArcLength(120, 'mnt')
spat_res=ArcLength(1, 'mnt')
spat_ext = ff.round_coord_to_res(ArcLength(max_spat_ext.base * size_factor), spat_res, high=True)
# spat_ext = ff.round_coord_to_res(ArcLength(max_spat_ext.base * 1.1), spat_res, high=True)
temp_res=Time(1, 'ms')
temp_ext=Time(500, 'ms')

orientation = ArcLength(90, 'deg')
temp_freq = TempFrequency(4)
spat_freq_x = SpatFrequency(2)
spat_freq_y = SpatFrequency(0)

st_params = do.SpaceTimeParams(
	spat_ext, spat_res, temp_ext, temp_res,
	array_dtype='float32'
	)

stim_params = do.GratingStimulusParams(
    spat_freq_x, temp_freq,
    orientation=orientation,
    contrast=do.ContrastValue(0.4)
)
# -
# +
stim = stimulus.mk_sinstim(st_params, stim_params)
print(f'predicted size (MB): {stim.nbytes / (1000*1000) * (1.1/size_factor)**2}')
# -



# +
# stim.nbytes / (1000*1000) * (1.1/0.5)**2
# -
# +
stim_test_path = Path('~/Downloads/stim_test_file.npy').expanduser()
print(stim_test_path)
np.save(stim_test_path.expanduser(), stim)
# -
# +
print(f'Actual size: {stim_test_path.stat().st_size / (1000*1000)}')
# -

# +
demo_lgn = lgn.mk_lgn_layer(lgn_params, spat_res)
# -
# +
from collections import Counter
# sorted((c.spat_filt.key for c in demo_lgn.cells))
Counter((c.spat_filt.key for c in demo_lgn.cells))
# -


# ## Saving and Caching

# +
signature = stimulus.mk_stim_signature(st_params, stim_params)
print(signature)
# -
# +
new_st_params, new_stim_params = stimulus.mk_params_from_stim_signature(signature)

print(st_params)
print(new_st_params)
print(new_st_params == st_params)
# -

# +
from dataclasses import replace

mod_st_params = replace(st_params, temp_ext=Time(534.5))

stimulus.mk_stim_signature(mod_st_params, stim_params)
# -
st_params.asdict_() == st_params.asdict_()
stim_params.asdict_()

hash(stim_params)

# +
multi_stim_params = stimulus.mk_multi_stimulus_params(
	do.MultiStimulusGeneratorParams(
		spat_freqs=[2,4], temp_freqs=[1,2], orientations=[0, 90],
	    spat_freq_unit='cpd', temp_freq_unit='hz', ori_arc_unit='deg',
	    )
	)
# -
# +
len(multi_stim_params)
# -
# +
for sp in multi_stim_params:
	print(sp.spat_freq.cpd, sp.temp_freq.hz, sp.orientation.deg)
# -

# Reworking the signature code to preserve ints a floats

# +
st_params.spat_ext.value
# -
# +
signature = stimulus.mk_stim_signature(st_params, stim_params)
print(signature)
# -
# +
new_st_params, new_stim_params = stimulus.mk_params_from_stim_signature(signature)

print(st_params)
print(new_st_params)
print(new_st_params == st_params)
# -

# Basics

# +
signature = stimulus.mk_stim_signature(st_params, stim_params)
print(signature, len(signature), sep='\n\n')
# -
stim_params.contrast

# +
new_st_params, new_stim_params = stimulus.mk_params_from_stim_signature(signature)

print(st_params)
print(new_st_params)
print(new_st_params == st_params)
print(stim_params)
print(new_stim_params)
print(stim_params == new_stim_params)
# -


# ### Testing the caching


# +
data_dir = settings.get_data_dir()
# -

# List all stimulus cache

# +
pkl_files = data_dir.glob('STIMULUS*')
for f in pkl_files:
	print(f.name)
# -

# Make some stim caches ... but keep it small

# +
spat_ext=ArcLength(660, 'mnt')
spat_res=ArcLength(1, 'mnt')
# spat_ext = ff.round_coord_to_res(ArcLength(max_spat_ext.base * 1.1), spat_res, high=True)
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

st_params = do.SpaceTimeParams(
	spat_ext, spat_res, temp_ext, temp_res,
	array_dtype='float32'
	)

multi_stim_params = stimulus.mk_multi_stimulus_params(
	do.MultiStimulusGeneratorParams(
		spat_freqs=[2], temp_freqs=[4], orientations=[90],
		)
	)
# -

# Add to stimulus cache

# +
stimulus.mk_stimulus_cache(st_params, multi_stim_params)
# -

# Get a dict of or print out all cached stimuli

# +
stimulus.get_params_for_all_saved_stimuli()
# -
# +
stimulus.print_params_for_all_saved_stimuli()
# -

# Load a stimulus from the cache

# +
stim_array = stimulus.load_stimulus_from_params(st_params, multi_stim_params[0])

stim_array.shape
# -

# ## Slicing Stimulus

# +
sf = lgn.cells.spatial_filters['maffei73_2right']
# -


# +
sf_loc = do.RFLocation(
	ArcLength(5.342, 'mnt'),
	ArcLength(0, 'mnt')
	)

spat_slice = stimulus.mk_rf_stim_spatial_slice_idxs(
	st_params, sf.parameters.parameters, sf_loc)
print(spat_slice)
# -

# +
slice_range = spat_slice.x2 - spat_slice.x1
# -

# +
xc, yc = ff.mk_spat_coords(st_params.spat_res, sd=sf.parameters.max_sd())
spat_filt = ff.mk_dog_sf(xc, yc, sf.parameters)
# -
# +
print(spat_filt.shape[0], slice_range)
# -

# +
spat_slice.is_within_extent(st_params)
# -
# +
sliced_array = stim_array[spat_slice.y1:spat_slice.y2, spat_slice.x1:spat_slice.x2]
# -
# +
sliced_array.shape
# -

# +
spat_slice.x2 *= 10
# -
# +
# Should be, and is False
spat_slice.is_within_extent(st_params)
# -

# +
print(spat_filt.shape[0] == (spat_slice.y2-spat_slice.y1))
# -

# +
stim_slice_array = stimulus.mk_stimulus_slice_array(st_params, stim_array, spat_slice)
# -
# +
stim_slice_array.shape
# -


# # Convolution Tidy up and testing

# +
spat_ext=ArcLength(250, 'mnt')
spat_res=ArcLength(1, 'mnt')
# spat_ext = ff.round_coord_to_res(ArcLength(max_spat_ext.base * 1.1), spat_res, high=True)
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

st_params = do.SpaceTimeParams(
	spat_ext, spat_res, temp_ext, temp_res,
	array_dtype='float32'
	)

multi_stim_params = stimulus.mk_multi_stimulus_params(
	do.MultiStimulusGeneratorParams(spat_freqs=[4], temp_freqs=[2], orientations=[0])
	)
# -

# More functional and simply takes stimulus slice and RF objects

# +
# stim_params = multi_stim_params[0]
orientation = ArcLength(90, 'deg')
temp_freq = TempFrequency(4)
spat_freq_x = SpatFrequency(2)
spat_freq_y = SpatFrequency(0)

stim_params = do.GratingStimulusParams(
	spat_freq_x, temp_freq,
	orientation=orientation,
	# amplitude=stim_amp, DC=stim_DC,
	contrast=do.ContrastValue(contrast=0.3)
	)
# -


# +
stimulus.mk_save_stim_array(st_params, stim_params)
stimulus.print_params_for_all_saved_stimuli()
# -
# +
stim_array = stimulus.load_stimulus_from_params(st_params, stim_params)
stim_array.shape
# -


# +
sf = lgn.cells.spatial_filters['maffei73_2right']
tf = lgn.cells.temporal_filters['kaplan87']
# -

# +
# _all_filters = filters.get_filters(filters.get_filter_index())
# spatial_filters = _all_filters['spatial']
# temporal_filters = _all_filters['temporal']
# sf = spatial_filters['maffei73_2right']
# tf = temporal_filters['kaplan87']
# -

# +
# xc, yc = ff.mk_spat_coords(st_params.spat_res, spat_ext=st_params.spat_ext)
# spat_filt = ff.mk_dog_sf(xc, yc, sf)

# tc = ff.mk_temp_coords(st_params.temp_res, temp_ext = st_params.temp_ext)
# temp_filt = ff.mk_tq_tf(tc, tf)

xc, yc = ff.mk_spat_coords(st_params.spat_res, sd=sf.parameters.max_sd())
spat_filt = ff.mk_dog_sf(xc, yc, sf)

tc = ff.mk_temp_coords(st_params.temp_res, tau=tf.parameters.arguments.tau)
temp_filt = ff.mk_tq_tf(tc, tf)
# -
# +
# shouldn't be necessary anymore
# sf.source_data.resp_params.resolve()
# sf.source_data.resp_params = sf.source_data.resp_params.resolve()
# -
# +
sf_loc = do.RFLocation(
	ArcLength(0, 'mnt'),
	ArcLength(0, 'mnt')
	)
# sf_loc = do.RFLocation(
# 	ArcLength(5.342, 'mnt'),
# 	ArcLength(0, 'mnt')
# 	)

spat_slice = stimulus.mk_rf_stim_spatial_slice_idxs(
	st_params, sf.parameters.parameters, sf_loc)
print(spat_slice)
# -
# +
stim_slice = stimulus.mk_stimulus_slice_array(st_params,stim_array, spat_slice)
print(stim_slice.shape)
# -
# +
cell_max_f1 = correction.mk_actual_filter_max_amp(sf, tf, stim_params.contrast)
cell_max_f1_val = cell_max_f1.value
print(cell_max_f1_val)
# -

# +
def actual_f1_amplitude(
		full_corrected_conv, time_coords):

	spec, _ = est_f1.gen_fft(full_corrected_conv, time=time_coords, align_freqs=True)
	spec = abs(spec)

	# indices and integer temp_freqs should align (as using "align_freqs=True")
	# this is the F1 amplitude
	actual_amplitude = spec[int(stim_params.temp_freq.hz)]

	return actual_amplitude
# -

# +
target_max_f1_amp = do.LGNF1AmpMaxValue(90, contrast=stim_params.contrast)
resp = convolve.mk_single_sf_tf_response(
	st_params, sf, tf, spat_filt, temp_filt,
	stim_params,
	# stim_array,
	stim_slice,
	filter_actual_max_f1_amp=cell_max_f1_val,
	target_max_f1_amp=target_max_f1_amp,
	rectified=True
	)
# -

# +
estimated_amplitude = resp.adjustment_params.joint_response.amplitude
if (max_f1_factor := resp.adjustment_params.max_f1_adj_factor):
    estimated_amplitude *= max_f1_factor

full_time_coord = ff.mk_temp_coords(st_params.temp_res, temp_ext = st_params.temp_ext)
resp_amp_error = (
	( estimated_amplitude - actual_f1_amplitude(resp.response, full_time_coord))
	/ estimated_amplitude
	)
print(f'{resp_amp_error:.2%} ... ({estimated_amplitude})')
# -
# +
px.line(resp.response).show()
# -


# # Poisson Spike Generation

# +
br2 = convolve.bn
# -


# +
spikes = convolve.mk_response_poisson_spikes(st_params, resp.response)
# -
# +
spikes.spike_trains()
# -

# ## Make spike trains for multiple LGN cells at once?

# +
input_responses = (resp.response, resp.response*100)
# -
# +
# alter the phase of one of the resposne
input_responses = (resp.response, np.roll(resp.response, -75)*100)
# -
# +
(
	go.Figure()
	.add_scatter(y=input_responses[0])
	.add_scatter(y=input_responses[1])
	.show()

)
# -
# +
spikes = convolve.mk_response_poisson_spikes(
	st_params, input_responses)
# -
# +
spike_times = spikes.spike_trains()
print(len(spike_times[0]), len(spike_times[1]))
# -

# +
plot.poisson_trials_rug(spike_times).show()
# -

# Final function for LGN spikes
# Takes a tuple of the response arrays ... maybe could be better contained?

# +
lgn_layer_resp = convolve.mk_lgn_response_spikes(st_params, input_responses)
# -


# # Making LGN layer and accessing Actual Max F1 values

# +
lgnparams = do.LGNParams(
	n_cells=30,
	# n_cells=3,
	orientation = do.LGNOrientationParams(ArcLength(30), 0.5),
	circ_var = do.LGNCircVarParams('naito_lg_highsf', 'naito'),
	F1_amps= do.LGNF1AmpDistParams(),
	spread = do.LGNLocationParams(2, 'jin_etal_on'),
	filters = do.LGNFilterParams(spat_filters='all', temp_filters='all')
	)
# -
# +
lgn_layer = cells.mk_lgn_layer(lgnparams, st_params.spat_res)
# -

# ## Testing Cell Record and pickling

# +
cell = lgn_layer.cells[0]
cell_record = cells.mk_lgn_cell_record(cell)

new_cell = cells.mk_cell_from_record(cell_record)

layer_record = cells.mk_lgn_layer_record(lgn_layer)

new_layer = cells.mk_lgn_layer_from_record(layer_record)
# -
# +
import pickle

with open('/tmp/test_cell.pkl', 'wb') as f:
	pickle.dump(cell, f)

with open('/tmp/test_lgn.pkl', 'wb') as f:
	pickle.dump(lgn_layer, f)

with open('/tmp/test_cell_rec.pkl', 'wb') as f:
	pickle.dump(cell_record, f)

with open('/tmp/test_lgn_rec.pkl', 'wb') as f:
	pickle.dump(layer_record, f)
# -
# +
with open('/tmp/test_lgn_rec.pkl', 'rb') as f:
	new_layer = cells.mk_lgn_layer_from_record(pickle.load(f))
# -
# +
print(
	new_layer.cells[0].spat_filt.parameters.cent.arguments
	==
	lgn_layer.cells[0].spat_filt.parameters.cent.arguments
)
# -

# +
actual_max_f1_amps = all_max_f1.mk_actual_max_f1_amps(stim_params)
# -
# +
list(actual_max_f1_amps.keys())[0]
# ('berardi84_5a', 'kaplan87')
# -

# +
test_cell = lgn_layer.cells[0]
# -
# +
test_cell.spat_filt.key
# -
# +
all_max_f1.spatial_filters['berardi84_6'].key
# -

# +
len(set(sf.key for sf in filters.spatial_filters.values()))
# -

# +
test = {sf.key: index_key for index_key, sf in filters.spatial_filters.items()}
# -
filters.spatial_filters[test[list(test.keys())[0]]]

# +
filters.reverse_spatial_filters[test_cell.spat_filt.key]
# -

# Test new function

# +
all_max_f1.get_cell_actual_max_f1_amp(test_cell, actual_max_f1_amps)
# -


# # Setting up LIF for V1 cell

# +
lif_params = do.LIFParams()
lif_params.mk_dict_with_units()
# -

# Initial network just with dummy spikes
# +
ntwk = lifv1.mk_lif_v1(2, lif_params)
# -
# +
ntwk.network.run(0.1*lifv1.bn.second)
# -
# +
ntwk.input_spike_generator._spike_time, ntwk.input_spike_generator._neuron_index
# -
# +
px.line(ntwk.membrane_monitor.v[0]).show()
# -

# +
ntwk.network.restore(ntwk.initial_state_name)
# -
# +
ntwk.membrane_monitor.v.shape
# -

# Update the spikes
# Note that the number of inputs has to stay the same (max of indices)
# +
px.line(ntwk.membrane_monitor.v[0]).show()
# -


# creating spike times and indices
# +
time_shift = 40
all_spike_times = (
		Time(np.arange(20)+time_shift, 'ms'),  # cell 1
		Time(np.arange(30)+time_shift, 'ms')   # cell 2
	)
# -
# +
spike_idxs, spike_times = lifv1.mk_input_spike_indexed_arrays(all_spike_times)
# -
# +
len(np.unique(spike_idxs))
# -
# +
spike_idxs, spike_times
# -
# +
ntwk.reset_spikes(spike_idxs, spike_times)
# -
# +
ntwk.network.restore(ntwk.initial_state_name)
ntwk.input_spike_generator.set_spikes(
	indices=spike_idxs,
	times=spike_times.ms * lifv1.bnun.msecond
	)
# -

# ## Replacing the lgn layer input Spike Generator object
# THis is necessary for doing synchrony, as the number of synchronous poisson inputs will vary
# from layer to layer

# Can just make a new ntwk object (which isn't really expensive??) and alter the number of inputs
# to that of the number of independent spiking elements in the LGN.

# +

# -


# # Trying new run function

# ## revisiting maximal required spat ext
# +
spat_res = ArcLength(1, 'mnt')
lgn_params = lgn.demo_lgnparams
# -
# +
max_spat_ext = stimulus.estimate_max_stimulus_spatial_ext_for_lgn(
	spat_res, lgn_params, n_cells=5000, safety_margin_increment=0.1)
print(max_spat_ext.mnt)
# -

# ## Setting up params

# +
spat_res=ArcLength(1, 'mnt')
spat_ext=ArcLength(660, 'mnt')
"Good high value that should include any/all worst case scenarios"
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

st_params = do.SpaceTimeParams(
	spat_ext, spat_res, temp_ext, temp_res,
	array_dtype='float32'
	)
# -
# +
# good subset of spat filts that are all in the middle in terms of size
subset_spat_filts = [
	'berardi84_5a', 'berardi84_5b', 'berardi84_6', 'maffei73_2mid',
	'maffei73_2right', 'so81_2bottom', 'so81_5', 'soodak87_1'
]
# -
# +
stimulus.print_params_for_all_saved_stimuli()
# -
# +
multi_stim_params = do.MultiStimulusGeneratorParams(
	spat_freqs=[0.8], temp_freqs=[4], orientations=[90], contrasts=[0.4]
	)
lgn_params = do.LGNParams(
	n_cells=30,
	orientation = do.LGNOrientationParams(ArcLength(0), circ_var=0.5),
	circ_var = do.LGNCircVarParams('naito_lg_highsf', 'naito'),
	spread = do.LGNLocationParams(2, 'jin_etal_on'),
	filters = do.LGNFilterParams(spat_filters='all', temp_filters='all'),
	F1_amps = do.LGNF1AmpDistParams()
	)
lif_params = do.LIFParams()
# -

# +
sim_params = do.SimulationParams(
	n_simulations=1,
	space_time_params=st_params,
	multi_stim_params=multi_stim_params,
	lgn_params=lgn_params,
	lif_params = lif_params
	)
# -


# ### Running Simulation

# +
results = run.run_simulation(sim_params)
# -
# +
results.params.lif_params
# -
# +
results.lgn_layers.keys()
len(list(results.lgn_layers.values())[0])
# -
# +
test_result = list(results.results.values())[0]
len(test_result[0].lgn_spikes.value)
# -

# ### Saving Results (?)

# +
test_dir = Path('/home/ubuntu/lif_hws/work/results_data')
run.save_simulation_results(
		results_dir = test_dir,
		sim_results = results,
		comments = 'test run'
	)
# -


# #### Random Dev Tests

# ##### testing new lgn trials

# +
response_pulse = np.ones(400)*50
response_arrays = (
		np.r_[np.zeros(100), response_pulse, np.zeros(500)],
		np.r_[np.zeros(300), response_pulse, np.zeros(300)],
		np.r_[np.zeros(500), response_pulse, np.zeros(100)],
		np.r_[np.zeros(600), response_pulse],
	)
# -
# +
lgn_resp = convolve.mk_lgn_response_spikes(
		st_params,
		response_arrays, n_trials = None
	)
# -
# +
fig = go.Figure()
for i, c in enumerate(lgn_resp.cell_spike_times):
	fig.add_scatter(
		y=i * np.ones_like(c.value),
		x=c.ms,
		mode='markers'
		)

fig.show()
# -
# +
n_trials = 9
lgn_resp = convolve.mk_lgn_response_spikes(
		st_params,
		(response_arrays[0],), n_trials = n_trials
	)
# -
# +
colors = ['red', 'green', 'blue', 'magenta']
fig = go.Figure()
for t, l in enumerate(lgn_resp):
	for i, c in enumerate(l.cell_spike_times):
		fig.add_scatter(
			y=(i * np.ones_like(c.value)) + (len(response_arrays) * t),
			x=c.ms,
			mode='markers',
			legendgroup = f'cell {i}',
			marker_color=colors[i]
			)

fig.show()
# -
# +
lgn_resp[0].cell_spike_times
# -
# +
len(lgn_resp)
test = lgn_resp[0]
# -
# +
print(
	len(test.cell_rates),
	len(test.cell_spike_times)
	)
# -
# +
lgn_resp[0].cell_spike_times[0] == lgn_resp[1].cell_spike_times[0]
# -


# ##### Testing Input spike indexed arrays

# test input spike indexed arrays function

# +
n_trials = 10
response_pulse = np.ones(400)*50
response_arrays = (
		np.r_[np.zeros(100), response_pulse, np.zeros(500)],
		np.r_[np.zeros(300), response_pulse, np.zeros(300)],
		np.r_[np.zeros(500), response_pulse, np.zeros(100)],
		np.r_[np.zeros(600), response_pulse],
	)
# -
# +
lgn_resp = convolve.mk_lgn_response_spikes(
		st_params,
		response_arrays, n_trials = n_trials
	)
# -

# +
spike_idxs, spike_times = (
	lifv1
	.mk_input_spike_indexed_arrays(lgn_response=lgn_resp)
	)
# -
# +
spike_idxs
spike_times
# -
# +
v1_synapse_idxs = cells.mk_repeated_v1_indices_for_inputs_for_all_lgn_and_trial_synapses(
		n_inputs=len(response_arrays), n_trials=n_trials)
# -
# +
fig = px.scatter(
	x=spike_times.ms,
	y=spike_idxs,
	color=[px.colors.qualitative.D3[v1_synapse_idxs[i_lgn]] for i_lgn in spike_idxs]
	)
fig.show()
# -

# test V1
# ... run a v1 simulation (can just use LGN response from above)

# +

# -




# ## Getting the V1 cell firing!

# Probelem is no firing from the V1 cell!
# Guessing that it's not enough EPSC current ...
# ... so ensuring that the total current (`EPSC x n_inputs`) is kept constant at `~2.5nA`

# ### Setup

# utility function for saving figs on the server

# +
def save_plotly_fig_tmp(fig):
	"""For graphs on a remote server - save to tmp to view through ssh later

	Write plotly figure to html with plotlyjs as a CDN.
	Files are written to `/tmp` with title if available and an incrememnted number.
	"""

	fig_dir = Path('/tmp')

	all_figure_files = fig_dir.glob('*.html')
	fig_numbers = sorted(
		[
			int(
				re.findall(r'.*fig_(\d{1,3}).html', f.as_posix())[0]
				)
			for f in all_figure_files]
		)
	next_number = fig_numbers[-1] + 1

	title_text = (fig.layout.title.text if fig.layout.title.text else '')
	fig_path = fig_dir / f'{title_text}_fig_{next_number}.html'

	fig.write_html(file=fig_path, include_plotlyjs='cdn')
# -

# basic fundamental space time params

# +
spat_res=ArcLength(1, 'mnt')
spat_ext=ArcLength(660, 'mnt')
"Good high value that should include any/all worst case scenarios"
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

st_params = do.SpaceTimeParams(
	spat_ext, spat_res, temp_ext, temp_res,
	array_dtype='float32'
	)
# -

# spatial filter subset

# +
# good subset of spat filts that are all in the middle in terms of size
subset_spat_filts = [
	'berardi84_5a', 'berardi84_5b', 'berardi84_6', 'maffei73_2mid',
	'maffei73_2right', 'so81_2bottom', 'so81_5', 'soodak87_1'
]

# subset_spat_filts = 'all'
# -

# quick check of available filters
# +
stimulus.print_params_for_all_saved_stimuli()
# -

# simulation params

# +
multi_stim_params = do.MultiStimulusGeneratorParams(
	spat_freqs=[1], # gets good coherent response from ensemble of LGN cells
	temp_freqs=[4],
	orientations=[90],
	contrasts=[0.4]
	)
lgn_params = do.LGNParams(
	n_cells=30,
	orientation = do.LGNOrientationParams(ArcLength(90), circ_var=0.99),
	circ_var = do.LGNCircVarParams('naito_lg_highsf', 'naito'),
	spread = do.LGNLocationParams(ratio=2, distribution_alias='jin_etal_on'),
	filters = do.LGNFilterParams(
		spat_filters=subset_spat_filts,
		temp_filters='all'),
	F1_amps = do.LGNF1AmpDistParams()
	)
lif_params = do.LIFParams(
	total_EPSC=3.5
	)
# lif_params.mk_dict_with_units(n_inputs=lgn_params.n_cells)
# -

# +
sim_params = do.SimulationParams(
	n_simulations=1,
	space_time_params=st_params,
	multi_stim_params=multi_stim_params,
	lgn_params=lgn_params,
	lif_params = lif_params
	)
# -

# ### Running Simulation

# +
results = run.run_simulation(sim_params)
# -


# ### Results analysis

# Extract results

# +
# get first simulation result (first stim, first simulation)
rks = list(results.keys())
result = results[rks[0]][0]
result.keys()
# -

# Plotting the LGN spiking rates


# LGN rates

# +
fig = go.Figure()
for i, r in enumerate(result['lgn_responses'].cell_rates):
	fig.add_scatter(y=r, mode='lines', name=f'cell {i}')
fig.update_layout(title='LGN_cell_rates')
# fig.show()
save_plotly_fig_tmp(fig)
# -

# average LGN rate
# +
avg_lgn_rate = np.mean(result['lgn_responses'].cell_rates, axis=0)
fig = px.line(y=avg_lgn_rate, title='average_lgn_rate')
save_plotly_fig_tmp(fig)
# -

# Membrane Potential

# +
membrane_potential = result['membrane_potential']
fig = px.line(membrane_potential, title='membrane_potential')
save_plotly_fig_tmp(fig)
# -

# lgn spikes

# +
spike_times = result['lgn_spikes']
fig = px.scatter(
	x=np.sort(spike_times.ms),
	y=np.ones_like(spike_times.ms)
	)
save_plotly_fig_tmp(fig)
# -
# +
fig = px.histogram(
	np.sort(spike_times.ms), nbins=50,
	title='lgn_spikes_histogram'
	)
save_plotly_fig_tmp(fig)
# -

# +
v1_spikes = result['spikes']
v1_spikes.shape
# -

# ### Pickling Prototyping

# +
import pickle

with open('/tmp/test.pkl', 'wb') as f:
	pickle.dump(sim_params, f)

example_lgn = cells.mk_lgn_layer(lgn_params, st_params.spat_res, do.ContrastValue(0.3))

with open('/tmp/test_lgn.pkl', 'wb') as f:
	pickle.dump(example_lgn, f)

example_lgn.cells[0].asdict_()

example_cell = example_lgn.cells[0]


with open('/tmp/test_lgn_cell.pkl', 'wb') as f:
	pickle.dump(example_cell, f)

example_cell.spat_filt.ori_bias_params = None
with open('/tmp/test_lgn_cell_no_ori_bias_params.pkl', 'wb') as f:
	pickle.dump(example_cell, f)

# -

# ### Multi LGN Layers
# so that can return to the same layer for different stimuli

# +
example_lgn = cells.mk_lgn_layer(lgn_params, st_params.spat_res, do.ContrastValue(0.3))
# -
# +
multi_example_lgn = [
	cells.mk_lgn_layer(lgn_params, st_params.spat_res, do.ContrastValue(0.3))
	for _ in range(1000)
	]

# -

# ### can the spike generator group of a network be changed after formation??


# ### Forcing RF Locs to be central

# +
example_lgn = cells.mk_lgn_layer(lgn_params, st_params.spat_res, force_central=True)
all(c.location.x.mnt == c.location.y.mnt == 0 for c in example_lgn.cells)
# -
# +
example_lgn = cells.mk_lgn_layer(lgn_params, st_params.spat_res, force_central=False)
all(c.location.x.mnt == c.location.y.mnt == 0 for c in example_lgn.cells)
# -
# +
# for c in example_lgn.cells:
# 	print(c.location.x.mnt, c.location.y.mnt)
# -

settings.simulation_params.spat_filt_sd_factor

# #### Greatest Spat Filt Ext?

# Which Spat FIlts are the largest ?
# +
for k, sf in sorted(
		cells.spatial_filters.items(),
		key = lambda x: x[1].parameters.max_sd().mnt
	):
	print(f'{k:<15} ... {sf.parameters.max_sd().mnt:.3f}')
# -

# Max Spat Ext required if all are centered?

# +
print(
	2
	*
	settings.simulation_params.spat_filt_sd_factor
	*
	max((sf.parameters.max_sd().mnt for sf in cells.spatial_filters.values()) )
)
# -
# +
stimulus.print_params_for_all_saved_stimuli()
# -

# If all RFs are centered, at the moment, max required is 475.2 mnts of spat_ext ... quite large!
# ... getting into 1G for `1000ms` of stimulus.

# Could limit the spat filts to the smaller ones:

# +
selected_spat_filts = (
	"maffei73_2right", "so81_2bottom", "berardi84_6", "soodak87_3", "maffei73_2mid", "so81_5",
	)
# -
# +
lgn_params_selected_sfs = do.LGNParams(
	n_cells=20,
	orientation = do.LGNOrientationParams(ArcLength(0), circ_var=0.5),
	circ_var = do.LGNCircVarParams('naito_lg_highsf', 'naito'),
	spread = do.LGNLocationParams(2, 'jin_etal_on'),
	filters = do.LGNFilterParams(
		spat_filters=selected_spat_filts,
		temp_filters='all'),
	F1_amps = do.LGNF1AmpDistParams()
	)
# -

# For worst case scenario

# +
stimulus.estimate_max_stimulus_spatial_ext_for_lgn(spat_res, lgn_params_selected_sfs, n_cells=5000)
# stimulus.estimate_max_stimulus_spatial_ext_for_lgn(spat_res, lgn_params, n_cells=1000)
# -

# ... `344 mnts`

# For all centered
# +
max_spat_ext = (
	2
	*
	settings.simulation_params.spat_filt_sd_factor
	*
	max(
		(sf.parameters.max_sd().mnt
			for key, sf in cells.spatial_filters.items()
			if key in lgn_params_selected_sfs.filters.spat_filters
			) )
)
# -

# ... `238 mnts`.
# Where `250 mnts`, `1000 ms` provides `256M` sized stimuli ... workable for testing


# ## Actual Run of a limited spatial extent


# Selecting only these spatial filters reduces the spatial extent necessary, especially if
# they're all centered.

# +
selected_spat_filts = (
	"maffei73_2right", "so81_2bottom", "berardi84_6", "soodak87_3", "maffei73_2mid", "so81_5",
	)
# -
# +
lgn_params = do.LGNParams(
	n_cells=20,
	orientation = do.LGNOrientationParams(ArcLength(0), circ_var=0.5),
	circ_var = do.LGNCircVarParams('naito_lg_highsf', 'naito'),
	spread = do.LGNLocationParams(2, 'jin_etal_on'),
	filters = do.LGNFilterParams(
		spat_filters=selected_spat_filts,
		temp_filters='all'),
	F1_amps = do.LGNF1AmpDistParams()
	)
# -

# +
spat_ext=ArcLength(250, 'mnt')
spat_res=ArcLength(1, 'mnt')
# spat_ext = ff.round_coord_to_res(ArcLength(max_spat_ext.base * 1.1), spat_res, high=True)
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

st_params = do.SpaceTimeParams(
	spat_ext, spat_res, temp_ext, temp_res,
	array_dtype='float32'
	)

# Stimulus parameters
multi_stim_params = do.MultiStimulusGeneratorParams(
		spat_freqs=[2], temp_freqs=[4], orientations=[90],
	)

lif_params = do.LIFParams()
# -
# +
sim_params = do.SimulationParams(
	n_simulations=1,
	space_time_params=st_params,
	multi_stim_params=multi_stim_params,
	lgn_params=lgn_params,
	lif_params = lif_params
	)
# -
# +
results = run.run_simulation(
	sim_params,
	force_central_rf_locations=True
	)
# -

# +
new_graph = plot.spat_filt_fit(
	filters.spatial_filters['so81_5'],
	hi_res_fit_only=False,
	normalise_magnitude=True,
	min_freq_bound=SpatFrequency(0.01), max_freq_bound=SpatFrequency(6)
	)
new_graph.show()
# -
new_graph.update_traces(name='hello').data[0]

plot.multi_spat_filt_fit(
	list(filters.spatial_filters.values()),
	share_freq_bounds=True
	).show()

reload(plot)

# ### Results analysis

# +
rks = list(results.keys())
result = results[rks[0]][0]
result.keys()
# -

# Plotting the LGN spiking rates

# +
fig = go.Figure()
for i, r in enumerate(result['lgn_responses'].cell_rates):
	fig.add_scatter(y=r, mode='lines', name=f'cell {i}')
fig.show()
# -


# ### LGN Dashboard

# Intended to help understand the result of a particular simulation

# Create an LGN layer

# +
spat_ext=ArcLength(250, 'mnt')
spat_res=ArcLength(1, 'mnt')
# spat_ext = ff.round_coord_to_res(ArcLength(max_spat_ext.base * 1.1), spat_res, high=True)
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

st_params = do.SpaceTimeParams(
	spat_ext, spat_res, temp_ext, temp_res,
	array_dtype='float32'
	)

lgn_params = do.LGNParams(
	n_cells=20,
	orientation = do.LGNOrientationParams(ArcLength(0), circ_var=0.5),
	circ_var = do.LGNCircVarParams('naito_lg_highsf', 'naito'),
	spread = do.LGNLocationParams(2, 'jin_etal_on'),
	filters = do.LGNFilterParams(spat_filters='all', temp_filters='all'),
	F1_amps = do.LGNF1AmpDistParams()
	)
# -
# +
lgn = cells.mk_lgn_layer(lgn_params, st_params.spat_res, force_central=False)
# -

# #### Spat Filt locations and chapes

# Locations
# +
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
# -
# +
lgn_cell_locations(lgn).show()
# -
# +
# cell_loc_params = []

fig=lgn_cell_locations(lgn)

for c in lgn.cells:
	x_loc, y_loc = c.location.x.mnt, c.location.y.mnt
	h_sd, v_sd = (
		c.oriented_spat_filt_params.parameters.cent.arguments.h_sd.mnt,
		c.oriented_spat_filt_params.parameters.cent.arguments.v_sd.mnt
		# c.spat_filt.parameters.cent.arguments.h_sd.mnt,
		# c.spat_filt.parameters.cent.arguments.v_sd.mnt
		)

	fig.add_shape(
		type="circle",
		xref="x", yref="y",
		x0=x_loc-(2*h_sd), x1=x_loc+(2*h_sd),
		y0=y_loc-(2*v_sd), y1=y_loc+(2*v_sd),
		line_color="black",
		)
fig.show()

	# cell_loc_params.append({'x':x_loc, 'y':y_loc, 'h': 2*h_sd, v_sd})
# -
# +
def lgn_sf_locations_and_shapes(lgn_layer: do.LGNLayer):

	x_locs = [c.location.x.mnt for c in lgn_layer.cells]
	y_locs = [c.location.y.mnt for c in lgn_layer.cells]

	fig = (
		px
		.scatter(x=x_locs, y=y_locs)
		.update_yaxes(scaleanchor = "x", scaleratio = 1)
		.update_layout(xaxis_constrain='domain', yaxis_constrain='domain')
		)

	for c in lgn.cells:
		x_loc, y_loc = c.location.x.mnt, c.location.y.mnt
		h_sd, v_sd = (
			c.oriented_spat_filt_params.parameters.cent.arguments.h_sd.mnt,
			c.oriented_spat_filt_params.parameters.cent.arguments.v_sd.mnt
			# c.spat_filt.parameters.cent.arguments.h_sd.mnt,
			# c.spat_filt.parameters.cent.arguments.v_sd.mnt
			)

		x0=x_loc-(2*h_sd)
		x1=x_loc+(2*h_sd)
		y0=y_loc-(2*v_sd)
		y1=y_loc+(2*v_sd)

		fig.add_shape(
			type="circle",
			xref="x", yref="y",
			x0=x0, x1=x1,
			y0=y0, y1=y1,
			line_color="black",
			)

	return fig
# -
# +
fig = lgn_sf_locations_and_shapes(lgn)
# -
# +
for i, s in enumerate(fig.layout.shapes):
	if i == 5:
		s.line.color = 'red'

# -


# Sorting LGN cells
# +
for c in sorted(lgn.cells, key=lambda c: c.location.y.mnt, reverse=True):
	print(f'{c.location.x.mnt:.2f}, {c.location.y.mnt:.2f}')
	# print(filters.reverse_spatial_filters[c.spat_filt.key])
# -

# Rotate ellipses by using bezier paths??

# ... sighs ... just draw them out

# +
fig = go.Figure()
fig.update_layout(
	shapes=[
		# Quadratic Bezier Curves
		dict(
			type="path",
			path="M1,5 A 5 3 20 0 1 8 8",
			line_color="RoyalBlue",
			),
		])
fig.show()
# -
# +
x_center=0
y_center=0
a=1.
b =1.
t = np.linspace(0, 2*np.pi, 20)
x = x_center + a*np.cos(t)
y = y_center + b*np.sin(t)
path = f'M {x[0]}, {y[0]}'

for k in range(1, len(t)):
	path += f'L{x[k]}, {y[k]}'
print(path)
# -
# +
%%timeit
path = f'M {x[0]}, {y[0]}'
for k in range(1, len(t)):
	path += f'L{x[k]}, {y[k]}'
# -
# +
import numpy as np
import plotly.graph_objects as go

def ellipse_arc(
		x_center=0, y_center=0, a=1., b =1.,
		ori=0,
		start_angle=0, end_angle=2*np.pi,
		N=20,
		):
	t = np.linspace(start_angle, end_angle, N)
	x = x_center + a*np.cos(t + np.radians(ori))
	# y = y_center + b*np.sin(t)
	y = y_center + b*np.sin(t + np.radians(ori))
	path = f'M {x[0]}, {y[0]}'
	for k in range(1, len(t)):
		path += f'L{x[k]}, {y[k]}'
	path += ' Z'
	return path
# -
# +
fig = go.Figure()

# Create a minimal trace
fig.add_trace(go.Scatter(
	x=[0],
	y=[0.2],
	marker_size=0.1
	));

fig.update_layout(width =600, height=400,
	xaxis_range=[-5.2, 5.2],
	yaxis_range=[-3.2, 3.2],
	shapes=[
	# dict(type="path",
	# 	path= ellipse_arc(a=5, b=3, start_angle=-np.pi/6, end_angle=3*np.pi/2, N=60),
	# 	line_color="RoyalBlue"),
	dict(type="path",
		path = ellipse_arc(x_center=0, y_center=0, a= 1, b= 3, ori=95),
		# fillcolor="LightPink",
		line_color="Crimson")
	]
	);
fig.show()
# -
# +
# Define ellipse parameters
x0 = 0  # x-coordinate of center
y0 = 0  # y-coordinate of center
a = 3   # major axis length
b = 1   # minor axis length

fig = (
	go.Figure()
	.update_yaxes(scaleanchor = "x", scaleratio = 1)
	.update_layout(xaxis_constrain='domain', yaxis_constrain='domain')
	)

# Generate array of angles
phi = np.linspace(0, 2*np.pi, 100)
for th in np.linspace(0, 180, 7, endpoint=False):

	theta = np.radians(th)  # rotation angle in radians


	# Calculate Cartesian coordinates of ellipse points
	x = x0 + a*np.cos(phi)*np.cos(theta) - b*np.sin(phi)*np.sin(theta)
	y = y0 + a*np.cos(phi)*np.sin(theta) + b*np.sin(phi)*np.cos(theta)

	fig.add_scatter(x=x, y=y, mode='lines', name=f'{th}')

fig.show()
# -


# # RF Pairwise Distance scale

# Use the mean or median of the max rf size of all possible pairs

# +
sfilts, tfilts = cells.mk_filters(20, lgn_params.filters)
rf_dist_scale = cells.rflocs.mk_rf_locations_distance_scale(
		sfilts, ArcLength(1, 'mnt')
	)
cent_sds = [sfilt.parameters.cent.arguments.h_sd.mnt for sfilt in sfilts]
print(rf_dist_scale)
print(np.mean(cent_sds), np.median(cent_sds))
for sfilt in sorted(sfilts, key=lambda sf: sf.parameters.cent.arguments.h_sd.mnt):
	print(sfilt.parameters.cent.arguments.h_sd.mnt)
# -


# +
n=300
rf_dist_scales=[None for _ in range(n)]
sf_sds_medians=[None for _ in range(n)]
sf_sds_means=[None for _ in range(n)]
for i in range(n):
	print(i, end='\r')
	sfilts, tfilts = cells.mk_filters(20, lgn_params.filters)
	rf_dist_scale = cells.rflocs.mk_rf_locations_distance_scale(
			sfilts, ArcLength(1, 'mnt'),
			use_median_for_pairwise_avg=False
		)
	cent_sds = [sfilt.parameters.cent.arguments.h_sd.mnt for sfilt in sfilts]

	rf_dist_scales[i] = rf_dist_scale.mnt
	sf_sds_medians[i] = np.median(cent_sds)
	sf_sds_means[i] = np.mean(cent_sds)

print(np.mean(rf_dist_scales), np.median(rf_dist_scales))
# -
# +
fig = (
	go.Figure()
	.add_scatter(
		x=sf_sds_medians, y=rf_dist_scales, mode='markers', name='sf_sd_medians')
	.add_scatter(
		x=sf_sds_means, y=rf_dist_scales, mode='markers', name='sf_sd_means')
	.update_layout(
		title='Mean used for avg pairwise max RF Size',
		xaxis_title='spat filt sd average', yaxis_title='RF pairwise distance scale')
	)
fig.show()

px.histogram(rf_dist_scales, title='RF Dist Scale (Mean avg)').show()
# -
# +
n=300
rf_dist_scales=[None for _ in range(n)]
sf_sds_medians=[None for _ in range(n)]
sf_sds_means=[None for _ in range(n)]
for i in range(n):
	print(i, end='\r')
	sfilts, tfilts = cells.mk_filters(20, lgn_params.filters)
	rf_dist_scale = cells.rflocs.mk_rf_locations_distance_scale(
			sfilts, ArcLength(1, 'mnt'),
			use_median_for_pairwise_avg=True
		)
	cent_sds = [sfilt.parameters.cent.arguments.h_sd.mnt for sfilt in sfilts]

	rf_dist_scales[i] = rf_dist_scale.mnt
	sf_sds_medians[i] = np.median(cent_sds)
	sf_sds_means[i] = np.mean(cent_sds)

print(np.mean(rf_dist_scales), np.median(rf_dist_scales))
# -
# +
fig = (
	go.Figure()
	.add_scatter(
		x=sf_sds_medians, y=rf_dist_scales, mode='markers', name='sf_sd_medians')
	.add_scatter(
		x=sf_sds_means, y=rf_dist_scales, mode='markers', name='sf_sd_means')
	.update_layout(title='Median used for avg pairwise max RF Size',
		xaxis_title='spat filt sd average', yaxis_title='RF pairwise distance scale')
	)
fig.show()

px.histogram(rf_dist_scales, title='RF Dist Scale (Median avg)').show()
# -


# Yea ... just use the mean ... cleaner and probably more accurate.

# +
dist_factor_coords = [
	cells.rflocs.spat_filt_coord_at_magnitude_ratio(
		spat_filt=sf.parameters, target_ratio=0.2, spat_res=spat_res)
	for sf in filters.spatial_filters.values()
]
# -
# +
px.scatter(
	y=[dfc.mnt for dfc in dist_factor_coords],
	x=[2 * sf.parameters.cent.arguments.h_sd.mnt for sf in filters.spatial_filters.values()],
	).update_layout(
		yaxis_title="radius at which amplitude is 20% of max",
		xaxis_title="2 Std Devs of center Gaussian"
	).add_scatter(
		mode='lines',
		x=[0, 35],y=[0, 0.8*35], line_color='#888'
	).show()
# -


# +
from itertools import combinations_with_replacement, combinations
# -
# +
sfilts, tfilts = cells.mk_filters(20, lgn_params.filters)
rf_dist_scale = cells.rflocs.mk_rf_locations_distance_scale(
		sfilts, ArcLength(1, 'mnt')
	)
# -
# +
for sf in sfilts:
	print(filters.reverse_spatial_filters[sf.key])
# -
# +
sfsds = [sf.parameters.cent.arguments.array()[-1] for sf in sfilts]
# -
# +
sfsd_combs = list(combinations_with_replacement(sfsds, r=2))
len(sfsd_combs)
# -
# +
sfsd_combs = list(combinations(sfsds, r=2))
len(sfsd_combs)
# -
# ## Actual Run

# +
results = run.run_simulation(sim_params)
# -

# +
membrane_potentials = [res[0]['membrane_potential']  for res in results.values()]
# -
# +
px.line(membrane_potentials[0]).show()
# -

# +
spike_times = [res[0]['lgn_spikes'] for res in results.values()]
# -
# +
px.scatter(
	x=np.sort(spike_times[0].ms),
	y=np.ones_like(spike_times[0].ms)
	).show()
# -
# +
px.scatter(spike_times[0].ms, ).show()
# -

# +
lgn_responses: do.LGNLayerResponse = [res[0]['lgn_responses'] for res in results.values()][0]
# -
# +
fig = go.Figure()
for i, r in enumerate(lgn_responses.cell_rates):
	fig.add_scatter(y=r, mode='lines', name=f'cell {i}')
fig.show()
# -
# +
all_response_sum = sum(lgn_responses.cell_rates)
# -
# +
px.line(all_response_sum).show()
# -
# +
px.histogram(np.sort(spike_times[0].ms), nbins=50).show()
# -
# +
# plot.poisson_trials_rug()
# -

# Checking the stimulus

# +
stim_params = stimulus.mk_multi_stimulus_params(sim_params.multi_stim_params)
# -
# +
stimulus_array = stimulus.load_stimulus_from_params(
	sim_params.space_time_params, stim_params[0])
# -
# +
center_idx = stimulus_array.shape[0] // 2
px.line(stimulus_array[center_idx, center_idx, :]).show()
px.line(stimulus_array[center_idx, :, 0]).show()
px.line(stimulus_array[:, center_idx, 0]).show()
# -


















