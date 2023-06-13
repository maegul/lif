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
import lif.utils.settings as settings,
import lif.utils.exceptions as exc

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
# +
import brian2 as bn
from brian2 import (
	units as bnun,
	Network,
	defaultclock,
	PoissonGroup,
	PoissonInput
	)

# bn.prefs.codegen.target = 'numpy'
# -


# # Functions
# +
def mk_lgn_spikes(
		lgn_rates,
		total_n_lgn_cells
		):

	bn.start_scope()

	timed_rate_arrays = bn.TimedArray(
		lgn_rates * bn.Hz,
		dt = 1 * bnun.ms
		)

	cell_group = bn.PoissonGroup(total_n_lgn_cells, rates='timed_rate_arrays(t,i)')
	spikes = bn.SpikeMonitor(cell_group)

	bn.run(1 * bn.second)

	spike_times = spikes.spike_trains()
	total_n_lgn_spikes = 0
	for k, v in spike_times.items():
		total_n_lgn_spikes += v.size

	all_spike_idxs = np.empty(shape=total_n_lgn_spikes)
	all_spike_times = np.empty(shape=total_n_lgn_spikes)

	total_spike_count = 0
	for k, v in spike_times.items():
		additional_spike_count = v.size
		new_spike_count = total_spike_count + additional_spike_count

		all_spike_idxs[total_spike_count : new_spike_count] = k
		all_spike_times[total_spike_count : new_spike_count] = v / bn.second

		total_spike_count = new_spike_count

	return all_spike_idxs, all_spike_times
# -
# +
def run_v1_sims(
		n_trials,
		n_lgn_cells,
		total_n_v1_cells,
		n_sims,
		all_spike_idxs,
		all_spike_times,
		lgn_synapse_idxs,
		v1_synapse_idxs
		):
	eqs = '''
	dv/dt = (v_rest - v + (I/g_EPSC))/tau_m : volt
	dI/dt = -I/tau_EPSC : amp
	'''

	on_pre =    'I += EPSC'
	threshold = 'v>v_thres'
	reset =     'v = v_reset'

	lif_params_w_units = lif_params.mk_dict_with_units(n_inputs=n_lgn_cells)

	G = bn.NeuronGroup(
		N=total_n_v1_cells,
		model=eqs,
		threshold=threshold, reset=reset,
		namespace=lif_params_w_units,
		method='euler')

	G.v = lif_params_w_units['v_rest']

	n_synapses = n_lgn_cells * n_trials * n_sims

	PS = bn.SpikeGeneratorGroup(
		N=n_synapses,
		indices=all_spike_idxs,
		times=all_spike_times * bn.second,
		sorted=True)

	S = bn.Synapses(PS, G, on_pre=on_pre, namespace=lif_params_w_units)
	S.connect(i=lgn_synapse_idxs, j=v1_synapse_idxs)

	M = bn.StateMonitor(G, 'v', record=True)
	SM = bn.SpikeMonitor(G)

	IM = bn.StateMonitor(G, 'I', record=True)
	ntwk = Network([G, PS, S, M, IM, SM])
	ntwk.run(1 * bnun.second)

	return SM, M
# -


# # Multi Simulations in One run of Brian

# ## Params
# +
n_lgn_cells = 30
n_trials = 10
n_sims = 200
lgn_spike_rate = 61
lif_params = do.LIFParams()
# -
# +
total_n_lgn_cells = n_lgn_cells * n_trials * n_sims
lgn_rates = np.ones(shape=(1000, total_n_lgn_cells)) * lgn_spike_rate
print('Mem Size of LGN Rates', lgn_rates.nbytes / 10**6)
# -

# ## LGN Spikes
# +
# using a function
# all_spike_idxs, all_spike_times = mk_lgn_spikes(lgn_rates, total_n_lgn_cells)
# -
# +
bn.start_scope()

timed_rate_arrays = bn.TimedArray(
	lgn_rates * bn.Hz,
	dt = 1 * bnun.ms)

cell_group = bn.PoissonGroup(total_n_lgn_cells, rates='timed_rate_arrays(t,i)')
spikes = bn.SpikeMonitor(cell_group)

bn.run(1 * bn.second)
# -

# ## Prep LGN Spikes for V1 Model
# +
spike_times = spikes.spike_trains()
total_n_lgn_spikes = 0
for k, v in spike_times.items():
	total_n_lgn_spikes += v.size

all_spike_idxs = np.empty(shape=total_n_lgn_spikes)
all_spike_times = np.empty(shape=total_n_lgn_spikes)

total_spike_count = 0
for i, (k, v) in enumerate(spike_times.items()):
	additional_spike_count = v.size
	new_spike_count = total_spike_count + additional_spike_count

	all_spike_idxs[total_spike_count : new_spike_count] = k
	all_spike_times[total_spike_count : new_spike_count] = v / bn.second

	total_spike_count = new_spike_count
# -
# +
# all_spike_idxs[-35:]
# all_spike_times[-35:]
# all_spike_idxs[:55]
# all_spike_times[:55]
# px.scatter(y=all_spike_times+all_spike_idxs, x=all_spike_idxs).show()
# -

# ## Prep V1 synapse indices
# +
lgn_synapse_idxs = np.arange(total_n_lgn_cells)
total_n_v1_cells = n_trials * n_sims
v1_synapse_idxs = np.array(
	cells.mk_repeated_v1_indices_for_inputs_for_all_lgn_and_trial_synapses(
		n_trials = total_n_v1_cells,
		n_inputs = (n_lgn_cells)
		)
	)

# v1_synapse_idxs.size == lgn_synapse_idxs.size
# -
# +
print(lgn_synapse_idxs.nbytes / 10**6)
print(v1_synapse_idxs.nbytes / 10**6)
# px.scatter(x=lgn_synapse_idxs, y=v1_synapse_idxs).show()
# -
# +
# Using a function
# SM, M = run_v1_sims(
# 		n_trials = n_trials,
# 		n_lgn_cells = n_lgn_cells,
# 		total_n_v1_cells = total_n_v1_cells,
# 		n_sims = n_sims,
# 		all_spike_idxs = all_spike_idxs,
# 		all_spike_times = all_spike_times,
# 		lgn_synapse_idxs = lgn_synapse_idxs,
# 		v1_synapse_idxs = v1_synapse_idxs
# 		)
# -

# +
bn.start_scope()

eqs = '''
dv/dt = (v_rest - v + (I/g_EPSC))/tau_m : volt
dI/dt = -I/tau_EPSC : amp
'''

on_pre =    'I += EPSC'
threshold = 'v>v_thres'
reset =     'v = v_reset'

lif_params_w_units = lif_params.mk_dict_with_units(n_inputs=n_lgn_cells)

# N=50
# G = bn.NeuronGroup(
# 	N=N,
# 	model=eqs,
# 	threshold=threshold, reset=reset,
# 	namespace=lif_params_w_units,
# 	method='euler')
G = bn.NeuronGroup(
	N=total_n_v1_cells,
	model=eqs,
	threshold=threshold, reset=reset,
	namespace=lif_params_w_units,
	method='euler')

G.v = lif_params_w_units['v_rest']

n_synapses = n_lgn_cells * n_trials * n_sims

# idxs = (np.r_[tuple(np.ones(3)*i for i in range(10000))]).astype(int)
# # spk_times = (np.ones(shape=(3 * 10000)) * 0.5) * bn.second
# spk_times = np.r_[tuple( np.array([0.1, 0.3, 0.6]) for _ in range(10_000) )] * bnun.second
# assert (idxs.size == spk_times.size)

# PS = bn.SpikeGeneratorGroup(
# 	N=10000,
# 	indices=idxs,
# 	times=spk_times,
# 	# sorted=True
# 	)


# S = bn.Synapses(PS, G, on_pre=on_pre, namespace=lif_params_w_units)
# v1_syn_idxs = (np.r_[tuple(np.ones(10_000//N) * i for i in range(50))]).astype(int)

# S.connect(i=np.arange(10000), j=v1_syn_idxs)

PS = bn.SpikeGeneratorGroup(
	N=n_synapses,
	indices=all_spike_idxs,
	times=all_spike_times * bn.second,
	# sorted=True
	)

S = bn.Synapses(PS, G, on_pre=on_pre, namespace=lif_params_w_units)
S.connect(i=lgn_synapse_idxs, j=v1_synapse_idxs)

M = bn.StateMonitor(G, 'v', record=True)
SM = bn.SpikeMonitor(G)

IM = bn.StateMonitor(G, 'I', record=True)
bn.run(1 * bnun.second)
# ntwk = Network([G, PS, S, M, IM, SM])
# ntwk.run(1 * bnun.second)
# -

# +
print(len(SM.spike_trains().keys()))
# -


# # LGN Convolution Performance

# +
stimulus.print_params_for_all_saved_stimuli()
stimulus.get_params_for_all_saved_stimuli()
# -
# +
all_saved_stim = stimulus.get_params_for_all_saved_stimuli()
for st, stim in all_saved_stim.items():
	if st.spat_ext.mnt == 660:
		print(st, stim)
		st_params, stim_params = st, list(stim)[0]
# -
# +
lgn_params = do.LGNParams(
	n_cells=10,
	orientation = do.LGNOrientationParams(ArcLength(0), circ_var=0.5),
	circ_var = do.LGNCircVarParams('naito_lg_highsf', 'naito'),
	spread = do.LGNLocationParams(2, 'jin_etal_on'),
	filters = do.LGNFilterParams(spat_filters='all', temp_filters='all'),
	F1_amps = do.LGNF1AmpDistParams()
	)

lgn_layer = cells.mk_lgn_layer(
	lgn_params,
	spat_res=st_params.spat_res,
	contrast=do.ContrastValue(0.3),
	force_central=False)

actual_max_f1_amps = all_max_f1.mk_actual_max_f1_amps(stim_params=stim_params)
# -
# +
lif_params = do.LIFParams()
multi_stim_params = do.MultiStimulusGeneratorParams(
	spat_freqs=[1], # gets good coherent response from ensemble of LGN cells
	temp_freqs=[4],
	orientations=[90],
	contrasts=[0.3]
	)
sim_params = do.SimulationParams(
	n_simulations=2,
	space_time_params=st_params,
	multi_stim_params=multi_stim_params,
	lgn_params=lgn_params,
	lif_params = lif_params,
	n_trials = 3
	# n_trials = 10
	)
# -

# +
stim_array = stimulus.load_stimulus_from_params(st_params, stim_params)
# -
# +
cell_resp = run.mk_lgn_cell_response(
	lgn_layer.cells[0], sim_params, stim_array, actual_max_f1_amps, stim_params)
# -
# +
1.5 * 30 * 1000 / (3600)
# -

# ## Profile `mk_single_sf_tf_Response`
# +
cell = lgn_layer.cells[0]
params = sim_params
# stim_array = stim_array
actual_max_f1_amps = actual_max_f1_amps
stim_params = stim_params
# -
# +
xc, yc = ff.mk_spat_coords(
	params.space_time_params.spat_res,
	sd=cell.spat_filt.parameters.max_sd()
	)

spat_filt = ff.mk_dog_sf(
	x_coords=xc, y_coords=yc,
		dog_args=cell.oriented_spat_filt_params  # use oriented params
		)
	# Rotate array
spat_filt = ff.mk_oriented_sf(spat_filt, cell.orientation)


	# temporal filter array
tc = ff.mk_temp_coords(
		params.space_time_params.temp_res,
		tau=cell.temp_filt.parameters.arguments.tau
		)
temp_filt = ff.mk_tq_tf(tc, cell.temp_filt)

	# slice stimulus
spat_slice_idxs = stimulus.mk_rf_stim_spatial_slice_idxs(
		params.space_time_params, cell.spat_filt, cell.location)
stim_slice = stimulus.mk_stimulus_slice_array(
		params.space_time_params, stim_array, spat_slice_idxs)

	# convolve
actual_max_f1_amp = all_max_f1.get_cell_actual_max_f1_amp(cell, actual_max_f1_amps)
# -
# +
0.07 * 30 * 1000 / (3600)
# -
# +
cell_resp = convolve.mk_single_sf_tf_response(
	params.space_time_params, cell.spat_filt, cell.temp_filt,
	spat_filt, temp_filt,
	stim_params, stim_slice,
	filter_actual_max_f1_amp=actual_max_f1_amp.value,
	target_max_f1_amp=cell.max_f1_amplitude
	)
# -
# +
1.53 * 30 * 1000 / (3600)
# -
# +
%prun convolve.mk_single_sf_tf_response(params.space_time_params, cell.spat_filt, cell.temp_filt, spat_filt, temp_filt, stim_params, stim_slice, filter_actual_max_f1_amp=actual_max_f1_amp.value, target_max_f1_amp=cell.max_f1_amplitude )
# -
# +
convolve.mk_single_sf_tf_response(
	params.space_time_params,
	cell.spat_filt,
	cell.temp_filt,
	spat_filt,
	temp_filt,
	stim_params,
	stim_slice,
	filter_actual_max_f1_amp=actual_max_f1_amp.value,
	target_max_f1_amp=cell.max_f1_amplitude
	)
# -
# +
st_params =  params.space_time_params
sf =  cell.spat_filt
tf =  cell.temp_filt
spat_filt =  spat_filt
temp_filt =  temp_filt
stim_params =  stim_params
stim_slice =  stim_slice
contrast_params = None
filter_actual_max_f1_amp =  filter_actual_max_f1_amp=actual_max_f1_amp.value
target_max_f1_amp =  target_max_f1_amp=cell.max_f1_amplitude
rectified = True
# -
# +
# # Handle if max_f1 passed in or not
if (
		(target_max_f1_amp or filter_actual_max_f1_amp) # at least one
		and not
		(target_max_f1_amp and filter_actual_max_f1_amp) # but not both
		):
	raise ValueError('Need to pass BOTH target and actual max_f1_amp')

# # requrie that the spatial filter and the stimulus slice are the same size
# stim_slice also has temporal dimension (3rd), so take only first two
if not (stim_slice.shape[:2] == spat_filt.shape):
	raise exc.LGNError('Stimulus slice and spatial filter array are not the same shape')

# # spatial convolution
spatial_product = (spat_filt[..., np.newaxis] * stim_slice).sum(axis=(0, 1))

# # temporal convolution

# prepare temp buffer
# Doesn't actually help get a stable sinusoidal response
# ... leaving here just in case it's useful later
# sf_conv_amp = correction.mk_dog_sf_conv_amp(
#     freqs_x=stim_params.spat_freq_x,
#     freqs_y=stim_params.spat_freq_y,
#     dog_args=sf.parameters, spat_res=st_params.spat_res
#     )
# buffer_val = sf_conv_amp * stim_params.DC
# print(sf_conv_amp, buffer_val)
# temp_res_unit = st_params.temp_res.unit
# buffer_size = int(Time(200, 'ms')[temp_res_unit] / st_params.temp_res.value ) + 1
# buffer = np.ones(buffer_size) * buffer_val

# spatial_product_w_buffer = np.r_[buffer, spatial_product]
# take temporal extent of stimulus, as convolve will go to extent of stim+temp_filt
# resp: np.ndarray = convolve(
#     spatial_product_w_buffer, temp_filt
#     )[buffer_size : (stim_slice.shape[2]+buffer_size)]

resp: np.ndarray = convolve.convolve(spatial_product, temp_filt)[:stim_slice.shape[2]]

# # adjustment parameters
# for going from F1 SF and TF to convolution to accurate sinusoidal response
adj_params = correction.mk_conv_resp_adjustment_params(
	st_params, stim_params, sf, tf,
	contrast_params=contrast_params,
	filter_actual_max_f1_amp=filter_actual_max_f1_amp,
	target_max_f1_amp=target_max_f1_amp
	)

# # apply adjustment
true_resp = correction.adjust_conv_resp(resp, adj_params)

# # recification
if rectified:
	true_resp[true_resp < 0] = 0

results = do.ConvolutionResponse(response=true_resp, adjustment_params=adj_params)
# -

# ## Target line (spatial convolution down to temporal)

# +
%timeit sp1 = (spat_filt[..., np.newaxis] * stim_slice).sum(axis=(0, 1))
# ~1.5 s
# -
# +
print(spat_filt.shape)
print(stim_slice.shape)
# (477, 477)
# (477, 477, 1001)
# -
# +
%timeit sp1 = (spat_filt[..., np.newaxis] * stim_slice)
# 1.2s
# -
# +
%timeit sp1.sum(axis=(0,1))
# 0.163
# -


# +
%timeit sp1 = stim_slice * 1.1
# -

# ### Custom Code
# +
%%timeit
temp_conv = np.zeros(shape=stim_slice.shape[2])
for t in range(stim_slice.shape[2]):
	temp_conv[t] = np.sum(spat_filt * stim_slice[:,:,t])
# -
# 5.11s

# +
%%timeit
temp_conv = np.array([
	np.sum(spat_filt * stim_slice[:,:,t])
	for t in range(stim_slice.shape[2])
	])
# -
# 4.9s

# +
def sf_stim_sum(sf, stim):

	temp_conv = np.zeros(shape=stim.shape[2])
	for t in range(stim.shape[2]):
		running_sum = 0
		for i in range(sf.shape[0]):
			for j in range(sf.shape[1]):
				running_sum += sf[i,j] * stim[i,j,t]
		temp_conv[t] = running_sum

	return temp_conv
# -
# +
%time sp1 = sf_stim_sum(spat_filt, stim_slice)
# -
# +
from numba import njit, float64, float32
# -
# +
@njit(float64[:](float64[:,:], float32[:,:,:]))
def sf_stim_sum_jit(sf, stim):
	temp_conv = np.zeros(shape=stim.shape[2])
	for t in range(stim.shape[2]):
		running_sum = 0
		for i in range(sf.shape[0]):
			for j in range(sf.shape[1]):
				running_sum += sf[i,j] * stim[i,j,t]
		temp_conv[t] = running_sum

	return temp_conv
# -
# +
sp1 = sf_stim_sum_jit(spat_filt, stim_slice)
%timeit sp1 = sf_stim_sum_jit(spat_filt, stim_slice)
# -
# +
stim_slice.dtype
# -



In [14]: m,n = 4,5

In [15]: A = np.random.rand(m,n)

In [16]: B = np.random.rand(m,n)

In [17]: np.sum(np.multiply(A, B))
Out[17]: 5.1783176986341335

In [18]: np.tensordot(A,B, axes=((0,1),(0,1)))
Out[18]: array(5.1783176986341335)

In [22]: A.ravel().dot(B.ravel())
Out[22]: 5.1783176986341335

In [21]: np.einsum('ij,ij',A,B)
Out[21]: 5.1783176986341326

# +
%timeit sp2 = np.einsum('ij,ij...', spat_filt, stim_slice)
# 0.336s
# -
# +
%timeit sp1 = (spat_filt[..., np.newaxis] * stim_slice).sum(axis=(0, 1))
# 1.56s
# -
# +
1.56 / 0.336
# 4.64
# -
# +
np.all(sp1 == sp2)
# -


# +
from scipy.ndimage import convolve as spconv
# -
# +
sp3 = spconv(stim_slice, spat_filt[..., np.newaxis])
# -
# +
sp3.shape
# -



# +
import cython
import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# distributed with Numpy).
# Here we've used the name "cnp" to make it easier to understand what
# comes from the cimported module and what comes from the imported module,
# however you can use the same name for both if you wish.
cimport numpy as cnp

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
cnp.import_array()
# -
# +
%%cython -a
def sf_stim_sum_cyth(sf, stim, temp_conv):

	for t in range(stim.shape[2]):
		running_sum = 0
		for i in range(sf.shape[0]):
			for j in range(sf.shape[1]):
				running_sum += sf[i,j] * stim[i,j,t]
		temp_conv[t] = running_sum

	return temp_conv

# -
# +
temp_conv = np.zeros(shape=stim_slice.shape[2])
result = sf_stim_sum_cyth(spat_filt, stim_slice, temp_conv)
# -


# +
from importlib import reload
# -
# +
import sf_stim_prod
from sf_stim_prod import sf_stim_sum_cyth
reload(sf_stim_prod)
# -
# +
temp_conv = np.zeros(shape=stim_slice.shape[2])
result = sf_stim_prod.sf_stim_sum_cyth(
	spat_filt.astype('float32'),
	stim_slice, temp_conv.astype('float32')
	)
# -

# +
def sf_stim_sum2(sf, stim):

	temp_conv = np.zeros(shape=stim.shape[2])
	for t in range(stim.shape[2]):
		running_sum = 0
		for i in range(sf.shape[0]):
			for j in range(sf.shape[1]):
				running_sum += sf[i,j] * stim[i,j,t]
		temp_conv[t] = running_sum

	return temp_conv

# -
# +
sp1 = sf_stim_sum2(spat_filt, stim_slice)
%timeit sp1 = sf_stim_sum2(spat_filt, stim_slice)
# -


