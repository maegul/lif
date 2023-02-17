

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
spat_ext=ArcLength(500, 'mnt')
spat_res=ArcLength(1, 'mnt')
# spat_ext = ff.round_coord_to_res(ArcLength(max_spat_ext.base * 1.1), spat_res, high=True)
temp_res=Time(1, 'ms')
temp_ext=Time(500, 'ms')

st_params = do.SpaceTimeParams(
	spat_ext, spat_res, temp_ext, temp_res,
	array_dtype='float32'
	)

multi_stim_params = stimulus.mk_multi_stimulus_params(
	do.MultiStimulusGeneratorParams(
		spat_freqs=[4], temp_freqs=[2], orientations=[0],
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
	n_cells=3,
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
ntwk = lifv1.mk_lif_v1(20, lif_params)
# -
# +
ntwk.network.run(0.1*lifv1.bn.second)
# -
# +
px.line(ntwk.membrane_monitor.v[0]).show()
# -


# Update the spikes
# Note that the number of inputs has to stay the same (max of indices)
# +
px.line(ntwk.membrane_monitor.v[0]).show()
# -


# creating spike times and indices
# +
all_spike_times = (
		np.arange(20),  # cell 1
		np.arange(30)   # cell 2
	)
# -
# +
spike_idxs, spike_times = lifv1.mk_input_spike_indexed_arrays(all_spike_times)
# -
# +
spike_idxs, spike_times
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
multi_stim_params = do.MultiStimulusGeneratorParams(
	spat_freqs=[2], temp_freqs=[4], orientations=[90]
	)
lgn_params = do.LGNParams(
	n_cells=20,
	orientation = do.LGNOrientationParams(ArcLength(0), circ_var=0.5),
	circ_var = do.LGNCircVarParams('naito_lg_highsf', 'naito'),
	spread = do.LGNLocationParams(2, 'jin_etal_on'),
	filters = do.LGNFilterParams(spat_filters='all', temp_filters='all'),
	F1_amps = do.LGNF1AmpDistParams()
	)
lif_params = do.LIFParams()
# -
# stimulus.mk_multi_stimulus_params(multi_stim_params)


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


















