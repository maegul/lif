
from dataclasses import dataclass

from typing import Union, Tuple, Optional, overload, cast

import numpy as np
import pandas as pd
import brian2 as bn
from brian2 import (
	units as bnun,
	Network,
	defaultclock,
	PoissonGroup,
	PoissonInput
	)

from brian2.equations.equations import parse_string_equations
bn.__version__

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as psp

from ..lgn import cells

import lif.utils.data_objects as do
from lif.utils.units.units import Time



def mk_lif_v1(
		n_inputs: int,
		lif_params: do.LIFParams,
		n_trials: Optional[int] = None
		) -> do.LIFNetwork:

	# equations
	eqs = '''
	dv/dt = (v_rest - v + (I/g_EPSC))/tau_m : volt
	dI/dt = -I/tau_EPSC : amp
	'''

	on_pre =    'I += EPSC'
	threshold = 'v>v_thres'
	reset =     'v = v_reset'

	lif_params_w_units = lif_params.mk_dict_with_units(n_inputs=n_inputs)
	number_trials = (
			1
				if not n_trials  # IE, only 1 if number of trials not provided
				else n_trials
		)

	G = bn.NeuronGroup(
		N=number_trials,
		model=eqs,
		threshold=threshold, reset=reset,
		namespace=lif_params_w_units,
		method='euler')

	# Set initial potential to threshold
	G.v = lif_params_w_units['v_rest']

	# each LGN cell, is simply given an index in series irrespective of which
	# LGN layer or rather, trial it belongs to (ie, 0, 1, 2, 3, ... to total amount of LGN cells)
	# The task is to direct these LGN cells to the appropriate V1 cells during simulation
	# so that there are simply a set of trials, each with the same LGN cells going to different
	# V1 cells ... and not a complete mess
	# Thus, number of synapses is the number of LGN cells, including each repeat across all
	# the trials.
	n_synapses = n_inputs * number_trials

	# custom spike inputs
	# make dummy spikes for initial state

	dummy_spk_idxs = np.arange(n_synapses)
	dummy_spk_times = dummy_spk_idxs * 2 * bnun.msecond

	PS = bn.SpikeGeneratorGroup(
		N=n_synapses,
		indices=dummy_spk_idxs,
		times=dummy_spk_times,
		sorted=False
		)

	S = bn.Synapses(PS, G, on_pre=on_pre, namespace=lif_params_w_units)

	# each synapse is an LGN cell, that implicitly belongs to a particular trial
	# the trials are organised for the sake of the simulation by connecting the LGN cells
	# appropriately.
	# This is done mostly with the "V1" (or `j`) indices ... ie the targets/post-synaptic cells.
	# The organisation
	v1_synapse_idx = np.array(
		cells.mk_repeated_v1_indices_for_inputs_for_all_lgn_and_trial_synapses(
				n_inputs=n_inputs, n_trials=number_trials
			)
		)
	# v1_synapse_idx = np.r_[
	# 	tuple(
	# 		n_trial * np.ones(n_inputs, dtype=int)
	# 		for n_trial in range(number_trials)
	# 		)
	# 	]
	S.connect(i=np.arange(n_synapses), j=v1_synapse_idx)

	M = bn.StateMonitor(G, 'v', record=True)
	SM = bn.SpikeMonitor(G)

	IM = bn.StateMonitor(G, 'I', record=True)
	ntwk = Network([G, PS, S, M, IM, SM])

	network = do.LIFNetwork(
		network=ntwk,
		input_spike_generator=PS,
		spike_monitor=SM, membrane_monitor=M,
		initial_state_name='initial',
		n_trials=n_trials
		)

	# paranoid ... ensuring initial statename is accurate in brian and this LIF object
	network.network.store(network.initial_state_name)

	return network


def mk_multi_lif_v1(
		n_inputs: int,
		n_simulations: int,
		n_trials: Optional[int],
		lif_params: do.LIFParams,
		) -> do.LIFMultiNetwork:

	eqs = '''
	dv/dt = (v_rest - v + (I/g_EPSC))/tau_m : volt
	dI/dt = -I/tau_EPSC : amp
	'''

	on_pre =    'I += EPSC'
	threshold = 'v>v_thres'
	reset =     'v = v_reset'

	lif_params_w_units = lif_params.mk_dict_with_units(n_inputs=n_inputs)
	number_trials = (
			1
				if not n_trials  # IE, only 1 if number of trials not provided
				else n_trials
		)

	total_n_v1_cells = n_simulations * number_trials
	n_synapses = n_inputs * number_trials * n_simulations

	G = bn.NeuronGroup(
		N=total_n_v1_cells,
		model=eqs,
		threshold=threshold, reset=reset,
		namespace=lif_params_w_units,
		method='euler')

	G.v = lif_params_w_units['v_rest']

	dummy_spk_idxs = np.arange(n_synapses)
	dummy_spk_times = dummy_spk_idxs * 2 * bnun.msecond

	PS = bn.SpikeGeneratorGroup(
		N=n_synapses,
		indices=dummy_spk_idxs,
		times=dummy_spk_times,
		sorted=False
		)

	S = bn.Synapses(PS, G, on_pre=on_pre, namespace=lif_params_w_units)

	v1_synapse_idxs = np.array(
		cells.mk_repeated_v1_indices_for_inputs_for_all_lgn_and_trial_synapses(
			# Here, n_trials is now n_trials * n_simulations, as that's the total number of v1 cells
			n_trials = total_n_v1_cells,
			n_inputs = n_inputs
			)
		)
	S.connect(i=np.arange(n_synapses), j=v1_synapse_idxs)

	M = bn.StateMonitor(G, 'v', record=True)
	SM = bn.SpikeMonitor(G)

	IM = bn.StateMonitor(G, 'I', record=True)
	ntwk = Network([G, PS, S, M, IM, SM])
	network = do.LIFMultiNetwork(
		network=ntwk,
		input_spike_generator=PS,
		spike_monitor=SM, membrane_monitor=M,
		initial_state_name='initial',
		n_trials=number_trials,
		n_simulations=n_simulations
		)

	# paranoid ... ensuring initial statename is accurate in brian and this LIF object
	network.network.store(network.initial_state_name)

	return network



def mk_input_spike_indexed_arrays(
		lgn_response: Union[
			Tuple[Time[np.ndarray], ...],
			do.LGNLayerResponse,
			Tuple[do.LGNLayerResponse]
			]
		) -> Tuple[np.ndarray, Time[np.ndarray]]:

	all_spike_times: Tuple[Time[np.ndarray], ...]

	# single trial lgn layer response
	if isinstance(lgn_response, do.LGNLayerResponse):
		all_spike_times = lgn_response.cell_spike_times

	# tuple of multiple trial results
	elif (isinstance(lgn_response, tuple)) and (isinstance(lgn_response[0], do.LGNLayerResponse)):
		lgn_response = cast(Tuple[do.LGNLayerResponse,...], lgn_response)
		# flatten all trial lgn response spike trains into a single tuple of arrays
		all_spike_times = tuple(
				spike_times
				for response in lgn_response
					for spike_times in response.cell_spike_times
			)

	# just a tuple of cell's spikes
	# elif isinstance(lgn_response, tuple) and not (isinstance(lgn_response[0], do.LGNLayerResponse)):
	else:
		lgn_response = cast(Tuple[Time[np.ndarray], ...], lgn_response)
		all_spike_times = lgn_response

	n_inputs = len(all_spike_times)

	spike_idxs = np.r_[
			tuple(
				# for each input, array of cell number same length as number of spikes
				(
					np.ones(shape=all_spike_times[i].value.size)
					* i
				).astype(int)
				for i in range(n_inputs)
			)
		]

	spike_times: Time[np.ndarray] = Time(
		np.r_[
			tuple(spike_times.ms for spike_times in all_spike_times)
			],
		'ms'
		)

	return spike_idxs, spike_times

	# spk_intvl = np.abs(all_psn_inpt_spikes[:, 1:] - all_psn_inpt_spikes[:, 0:-1])
	# spk_intvl_within_dt_idx = (spk_intvl <= (defaultclock.dt))
	# n_spikes_within_dt = np.sum(spk_intvl_within_dt_idx)

	# # exclude the first spike from masking, as always included
	# spks_flat_without_multi = np.r_[
	# 	all_psn_inpt_spikes[:, 0],
	# 	all_psn_inpt_spikes[:,1:][~spk_intvl_within_dt_idx]
	# 	]

	# # check that total spikes is right amount
	# assert (
	# 	(all_psn_inpt_spikes.flatten().shape - n_spikes_within_dt)
	# 	==
	# 	spks_flat_without_multi.shape
	# 	)

	# spks_idxs_flat = np.r_[
	# 	np.arange(n_inputs),  # index for all the first spikes
	# 	spike_idxs[~spk_intvl_within_dt_idx]  # already excludes the first spikes
	# ]


