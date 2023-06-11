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
# +
import brian2 as bn
from brian2 import (
	units as bnun,
	Network,
	defaultclock,
	PoissonGroup,
	PoissonInput
	)

bn.prefs.codegen.target = 'numpy'
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
