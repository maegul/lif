
from dataclasses import dataclass

from typing import Union, Tuple

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

import lif.utils.data_objects as do
from lif.utils.units.units import Time


def mk_lif_v1(
		n_inputs: int,
		lif_params: do.LIFParams
		) -> do.LIFNetwork:

	# equations
	eqs = '''
	dv/dt = (v_rest - v + (I/g_EPSC))/tau_m : volt
	dI/dt = -I/tau_EPSC : amp
	'''

	on_pre =    'I += EPSC'
	threshold = 'v>v_thres'
	reset =     'v = v_reset'

	lif_params_w_units = lif_params.mk_dict_with_units()
	G = bn.NeuronGroup(
		1,
		eqs,
		threshold=threshold, reset=reset,
		namespace=lif_params_w_units,
		method='euler')

	# Set initial potential to threshold
	G.v = lif_params_w_units['v_rest']

	# custom spike inputs
	# make dummy spikes for initial state
	dummy_spk_idxs = np.arange(n_inputs)
	dummy_spk_times = dummy_spk_idxs * 2 * bnun.msecond
	PS = bn.SpikeGeneratorGroup(
		n_inputs,
		dummy_spk_idxs,
		dummy_spk_times,
		sorted=True)

	S = bn.Synapses(PS, G, on_pre=on_pre, namespace=lif_params.mk_dict_with_units())
	S.connect(i=np.arange(n_inputs), j=0)

	M = bn.StateMonitor(G, 'v', record=True)
	SM = bn.SpikeMonitor(G)

	IM = bn.StateMonitor(G, 'I', record=True)
	ntwk = Network([G, PS, S, M, IM, SM])

	network = do.LIFNetwork(
		network=ntwk,
		input_spike_generator=PS,
		spike_monitor=SM, membrane_monitor=M,
		initial_state_name='initial'
		)

	# paranoid ... ensuring initial statename is accurate in brian and this LIF object
	network.network.store(network.initial_state_name)

	return network


def mk_input_spike_indexed_arrays(
		lgn_response: Union[Tuple[Time[np.ndarray], ...], do.LGNLayerResponse]
		) -> Tuple[np.ndarray, Time[np.ndarray]]:

	if isinstance(lgn_response, do.LGNLayerResponse):
		all_spike_times = lgn_response.cell_spike_times
	else:
		all_spike_times = lgn_response

	n_inputs = len(all_spike_times)
	# intervals correspond to spike `1` (second spike) to the end
	spike_idxs = np.r_[
			tuple(
				# for each input, array of cell number same length as number of spikes
				(np.ones(shape=all_spike_times[i].value.size) * i)
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


