
from dataclasses import dataclass

from typing import Union, Tuple, Optional, overload, cast, Sequence
from collections import deque

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

from ..lgn import cells, spat_filt_overlap as sfo

import lif.utils.data_objects as do
from lif.utils.units.units import Time
import lif.utils.exceptions as exc



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

@overload
def mk_multi_lif_v1(
		n_inputs: int,
		n_cells: None,
		n_simulations: int,
		n_trials: Optional[int],
		lif_params: do.LIFParams,
		) -> do.LIFMultiNetwork: ...
@overload
def mk_multi_lif_v1(
		n_inputs: Sequence[int],
		n_cells: int,  # if using variable number inputs, this should be the number of "true" cells
		n_simulations: int,
		n_trials: Optional[int],
		lif_params: do.LIFParams,
		) -> do.LIFMultiNetwork: ...
def mk_multi_lif_v1(
		n_inputs: Union[int, Sequence[int]],
		n_cells: Optional[int],  # if using synchrony, this should be the number of "true" cells
		n_simulations: int,
		n_trials: Optional[int],
		lif_params: do.LIFParams,
		) -> do.LIFMultiNetwork:

	eqs = '''
	dv/dt = (v_rest - v + (I/g_EPSC))/tau_m : volt
	dI/dt = -I/tau_EPSC : amp
	'''

	# `N_incoming` is a built in variable in Brian:
	# 	number of incoming synapses to the post-synaptic neuron of each particular synapse
	threshold = 'v>v_thres'
	reset =     'v = v_reset'

	# False as relying on `N_incoming` in `on_pre` equation
	# which will work whether it is a constant across all V1s or variable
	# (due to variable amounts of overlapping regions between layers)

	# no n_cells provided, so rely on normalisation by the number of inputs
	# n_inputs should be int
	if (n_cells is None):
		on_pre =    'I += EPSC'
		# n_inputs should be int from typing overloads above
		lif_params_w_units = lif_params.mk_dict_with_units(n_inputs = n_inputs)  # type: ignore

		# `N_incoming` is a built in variable in Brian:
		# 	number of incoming synapses to the post-synaptic neuron of each particular synapse
		# on_pre =    'I += EPSC/N_incoming'
		# False as relying on `N_incoming` in `on_pre` equation
		# which will work whether it is a constant across all V1s or variable
		# (due to variable amounts of overlapping regions between layers)
		# lif_params_w_units = lif_params.mk_dict_with_units(n_inputs = False)

	# use the "True number"
	# n_inputs should be a sequence
	else:
		on_pre =    'I += EPSC'
		lif_params_w_units = lif_params.mk_dict_with_units(n_inputs = n_cells)

	# Older way of normalising the EPSC values manually (now just rely on N_incoming variable)
	# lif_params_w_units = lif_params.mk_dict_with_units(
	# 	n_inputs=(
	# 			n_inputs
	# 				if isinstance(n_inputs, int) else
	# 				False  # if sequence, than don't normalise EPSC
	# 			 )
	# 	)

	number_trials = (
			1
				if not n_trials  # IE, only 1 if number of trials not provided
				else n_trials
		)

	# one v1 cell for each trial of each LGNlayer
	total_n_v1_cells = n_simulations * number_trials

	if isinstance(n_inputs, int):
		n_synapses = n_inputs * number_trials * n_simulations

		v1_synapse_idxs = np.array(
			cells.mk_repeated_v1_indices_for_inputs_for_all_lgn_and_trial_synapses(
				# Here, n_trials is now n_trials * n_simulations, as that's the total number of v1 cells
				n_trials = total_n_v1_cells,
				n_inputs = n_inputs
				)
			)

	else:
		# sum of all n_inputs for each layer (as a Sequence) is same as n_inputs * n_simulations
		# ... when n_inputs is constant across all layers
		n_synapses = sum(n_inputs) * number_trials

		v1_synapse_idxs = np.array(
			cells.mk_repeated_v1_indices_for_inputs_for_all_lgn_and_trial_synapses(
				# Here, n_trials is number of trials for each layer
				n_trials = number_trials,
				# n_inputs, a sequence of ints, each int representing an lgn layer and its v1 cell
				n_inputs = n_inputs
				)
			)

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


def clean_sycnchrony_spike_times(
		spike_times: Time[np.ndarray],
		temp_ext: Time[float],
		simulation_temp_res: Optional[Time[float]] = None
		) -> Time[np.ndarray]:

	"""Remove spikes that are negative, past simulation time or too close to each other

	Args:
		temp_ext: temporal extent of the simulation
		simulation_temp_res:
			temporal resolution of the brian spiking simulation (not lgn stimulus)
			If not provided, rely on global var for brian defaultclock
	"""

	if not simulation_temp_res:
		sim_temp_res = Time(defaultclock.dt / bnun.msecond, 'ms')
	else:
		sim_temp_res = simulation_temp_res

	# sort spikes (necessary as likely to be unsorted)
	spks = np.sort(spike_times.ms)
	de_dup_spks = None  # placeholder should no de-duplication need to occur

	# jitter pushed spikes below 0?
	if np.any(spk_negative := spks<0):
		spks[spk_negative] *= -1  # rotate jitter around 0 (ie, make it positive)

	# jitter pushed spikes beyond simulation time?
	if np.any(spk_late := spks > temp_ext.ms):
		# rotate around simulation time as with negatives above
		spks[spk_late] -= (spks[spk_late] - temp_ext.ms)

	# any two spikes closer than the simulation resolution?
	if np.any(spk_dup_idxs := (np.abs( spks[1:] - spks[0:-1] ) <= (sim_temp_res.ms)) ):

		# do not include in final array
		de_dup_spks = np.r_[spks[0], spks[1:][~spk_dup_idxs]]


	managed_spike_times = Time(
			de_dup_spks
				if (de_dup_spks is not None) else
			spks,
			'ms'  # make sure using same unit throughout as above
		)

	return managed_spike_times




@overload
def mk_input_spike_indexed_arrays(
		lgn_response: Tuple[do.LGNLayerResponse],
		overlapping_region_map: Tuple[sfo.LGNOverlapMap, ...],
		synchrony_params: do.SynchronyParams,
		temp_ext: Time[float],
		n_layers: int,
		n_trials: int,
		n_cells: int,
		simulation_temp_res: Optional[Time[float]] = None,
		) -> Tuple[np.ndarray, Time[np.ndarray], Tuple[Time[np.ndarray], ...]]: ...
@overload
def mk_input_spike_indexed_arrays(
		lgn_response: Union[
				Tuple[Time[np.ndarray], ...],
				do.LGNLayerResponse,
				Tuple[do.LGNLayerResponse]
			],
		overlapping_region_map: None = None,
		synchrony_params: None = None,
		temp_ext: None = None,
		n_layers: None = None,
		n_trials: None = None,
		n_cells: None = None,
		simulation_temp_res: None = None,
		) -> Tuple[np.ndarray, Time[np.ndarray], Tuple[Time[np.ndarray], ...]]: ...
def mk_input_spike_indexed_arrays(
		lgn_response: Union[
				Tuple[Time[np.ndarray], ...],
				do.LGNLayerResponse,
				Tuple[do.LGNLayerResponse]
			],
		overlapping_region_map: Optional[Tuple[sfo.LGNOverlapMap, ...]] = None,
		synchrony_params: Optional[do.SynchronyParams] = None,
		temp_ext: Optional[Time[float]] = None,
		n_layers: Optional[int] = None,
		n_trials: Optional[int] = None,
		n_cells: Optional[int] = None,
		simulation_temp_res: Optional[Time[float]] = None,
		) -> Tuple[np.ndarray, Time[np.ndarray], Tuple[Time[np.ndarray], ...]]:

	all_spike_times: Tuple[Time[np.ndarray], ...]

	# single trial lgn layer response
	if isinstance(lgn_response, do.LGNLayerResponse):
		all_spike_times = lgn_response.cell_spike_times

	# tuple of multiple trial results BUT no overlap map for synchrony
	elif (
				(isinstance(lgn_response, tuple))
				and
				(isinstance(lgn_response[0], do.LGNLayerResponse))
				and
				((synchrony_params is None) or (not synchrony_params.lgn_has_synchrony))
			):
		lgn_response = cast(Tuple[do.LGNLayerResponse,...], lgn_response)
		# flatten all trial lgn response spike trains into a single tuple of arrays
		all_spike_times = tuple(
				spike_times
				for response in lgn_response
					for spike_times in response.cell_spike_times
			)

	# multiple trial results WITH overlap map for synchrony
	# ... idea being to duplicate spike times back to original source LGN cells
	# ... with jitter
	elif (
				(isinstance(lgn_response, tuple))
				and
				(isinstance(lgn_response[0], do.LGNLayerResponse))
				and  # IE - doing synchrony?
				(overlapping_region_map is not None)
				and
				((synchrony_params is not None) and synchrony_params.lgn_has_synchrony)
			):


		# new ...
		mk_jitter = lambda jitter, size: np.random.normal(loc=0, scale=jitter, size=size)

		# should be true because of overload
		n_layers = cast(int, n_layers)
		n_trials = cast(int, n_trials)
		n_cells = cast(int, n_cells)
		# technically covered by conditional above
		lgn_response = cast(Tuple[do.LGNLayerResponse], lgn_response)
		temp_ext = cast(Time[float], temp_ext)

		# for the complecting of layers and trials, indices of which is which for each trial-layer
		trial_layer_idxs = tuple(
			{'n_layer': n_layer, 'n_trial': n_trial}
			for n_layer in range(n_layers)
				for n_trial in range(n_trials)
			)

		# cell idxs for each overlapping region
		# nested ... first layer is lgn layers (n_simulations), second is overlapping regions
		# ... then, for each region, the cell idxs that contribute to it (or overlap there)
		overlapping_map_layer_cell_idxs = tuple(
				tuple(
						tuple(keys) for keys in layer_overlapping_map
					)
				for layer_overlapping_map in overlapping_region_map
			)

		# sequence for all "true" lgn cell inputs (flattened from all trial-layers)
		# "true" as currently all spikes are grouped into "overlapping regions", but this process
		# ... will duplicate and rearrange these spikes back into the original number of lgn cells
		# ... (ie, not the lgn cells overlapping regions)
		true_all_spike_times_with_synchrony: deque[Time[np.ndarray]]  = deque()

		# go through each trial-layer
		for trial_layer_idx, trial_layer in zip(trial_layer_idxs, lgn_response):
			# list of groups (ie deques) for each "true" lgn cell, that will collect all of the
			# ... spikes that are being duplicated and arranged to come from this cell
			layer_cell_spikes = [deque() for _ in range(n_cells)]

			# go through each overlapping region (grabbing both its idx and spike times)
			for overlapping_region_idx, spike_times in enumerate(trial_layer.cell_spike_times):
				# get the "true cell" idxs of the lgn cells that overlap in this overlapping region
				# ... these will be the cells to which the spikes will be duplicated.
				true_cell_idxs = (
					overlapping_map_layer_cell_idxs
						[trial_layer_idx['n_layer']]
							[overlapping_region_idx]
					)
				# This is the MAIN EVENT ...
				# ... duplicate the spikes from the overlapping region (adding jitter should duplicate)
				# ... then assign to each of the "true lgn" cells that overlap in this region
				for cell_idx in true_cell_idxs:
					layer_cell_spikes[cell_idx].append(
									spike_times.ms + mk_jitter(
										synchrony_params.jitter.ms, spike_times.value.size),
						)
			# go through each "true cell" and concatenate all of the spike time arrays
			# ... they're separate because various arrays of spikes have been assigned from all of
			# ... of the overlapping regions
			for cell_spikes in layer_cell_spikes:
				# concatenate all spike_times
				concatenated_spike_times = Time(
						np.r_[tuple(spike_times for spike_times in cell_spikes)],
						'ms'
					)
				# DE-DUPLICATE HERE!!
				managed_spike_times = clean_sycnchrony_spike_times(
						concatenated_spike_times, temp_ext, simulation_temp_res
					)
				true_all_spike_times_with_synchrony.append(managed_spike_times)


		# start_idxs, end_idxs = (
		# 	(start_idxs := np.arange(0, len(true_all_spike_times_with_synchrony), n_cells)),
		# 	(start_idxs + 30)
		# 	)

		# # check
		# if (
		# 			# final idx should be total len of spike_times sequence (ie n cells in all trial_layers)
		# 			(end_idxs[-1] != (n_layers*n_trials*n_cells))
		# 			or
		# 			# number of indices should be same as number of trial_layers (ie len of lgn_response)
		# 			(len(start_idxs) != len(lgn_response))
		# 			or
		# 			# number of indices should be same as number of trial_layers
		# 			(len(lgn_response) != (n_layers * n_trials))
		# 		):

		# 	raise exc.SimulationError("Trial, layer, cell indices don't match expected sizes")

		all_spike_times = tuple(true_all_spike_times_with_synchrony)

		# new_lgn_layer_spike_times = tuple(  # trial_layer_responses
		# 	# all spike times for each cell in trial_layer
		# 	tuple(all_spike_times[start_idx:end_idx])
		# 	for start_idx, end_idx in zip(start_idxs, end_idxs)
		# 	)

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
					i * np.ones(shape=all_spike_times[i].value.size)
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

	return spike_idxs, spike_times, all_spike_times

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


