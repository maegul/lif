# # Imports
# +
import shutil
import itertools
import copy
import time
from pathlib import Path
from typing import Any, Literal, Tuple, Union, Dict, List, Sequence, TypedDict, Callable, Optional, overload
import re
import datetime as dt
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
import multiprocessing as mp
from multiprocessing.pool import AsyncResult
# -


# # Utilities for handling multi simulation parameters

class MultiSimAgents(TypedDict):
	values: List[Any]
	attr_path: str


def set_sim_param(sim_params: do.SimulationParams, v: Any, attr_path: str):

	sim_params = copy.deepcopy(sim_params)
	sim_attribute = sim_params
	path_elements = attr_path.split('.')

	for i, element in enumerate(path_elements):
		try:
			new_attribute = getattr(sim_attribute, element)
			# if last element, don't assign, as it will be used to assign ...
			# ... but still need to check if final attribute exists
			if i < (len(path_elements) - 1):
				sim_attribute = new_attribute
		except AttributeError:
			raise ValueError(
				f'Attribute "{element}" (from path {attr_path}) does not exist on sim_params')

	setattr(sim_attribute, element, v)  # type: ignore

	return sim_params


@overload
def mk_all_sim_params(
		sim_params: do.SimulationParams,
		multi_sim_update_agents: Optional[Tuple[MultiSimAgents, ...]],
		return_combinations_only: Literal[False] = False
		) -> List[do.SimulationParams]: ...
@overload
def mk_all_sim_params(
		sim_params: do.SimulationParams,
		multi_sim_update_agents: Optional[Tuple[MultiSimAgents, ...]],
		return_combinations_only: Literal[True]
		) -> List[Tuple[Tuple[Any, str], ...]]: ...
def mk_all_sim_params(
		sim_params: do.SimulationParams,
		multi_sim_update_agents: Optional[Tuple[MultiSimAgents, ...]],
		return_combinations_only: bool = False
		) -> Union[List[do.SimulationParams], List[Tuple[Tuple[Any, str], ...]]]:
	"""

	Examples:

		* Using the return_combinations_only option ...

		```python
		multi_sim_params_ratios = MultiSimAgents(
			values=[1, 2, 3, 4, 10], attr_path='lgn_params.spread.ratio')

		multi_sim_params_cv = MultiSimAgents(
			values=[0.1, 0.2, 0.3], attr_path='lgn_params.orientation.circ_var')

		all_multi_params = (multi_sim_params_cv, multi_sim_params_ratios)

		all_combos = mk_all_sim_params(sim_params, all_multi_params, True)
		```
	"""
	# +

	if multi_sim_update_agents is None:
		return [sim_params]

	# repeat paths for each value
	# ie, for each value, put it in a tuple along with its custom path
	all_agents = [
			[(v, agent['attr_path']) for v in agent['values']]
			for agent in multi_sim_update_agents
		]
	# nest the sweeps ... ie cross product of all variations
	all_agent_combos: List[Tuple[Tuple[Any, str], ...]] = list(itertools.product(*all_agents))

	if return_combinations_only:
		return all_agent_combos

	# create new sim_params accordingly
	# sweep through all combinations of values, update sim_params, store in list
	all_sim_params = []
	for agent_combos in all_agent_combos:
		new_sim = copy.deepcopy(sim_params)
		for agent in agent_combos:
			new_sim = set_sim_param(new_sim, agent[0], agent[1])
		all_sim_params.append(new_sim)
	# -

	return all_sim_params


# class MultiSimAgents(TypedDict):
# 	values: List
# 	func: Callable


# def mk_all_sim_params(
# 		sim_params: do.SimulationParams,
# 		multi_sim_update_agents: Optional[Tuple[MultiSimAgents, ...]]
# 		) -> List[do.SimulationParams]:
# 	# +

# 	if multi_sim_update_agents is None:
# 		return [sim_params]

# 	# repeat funcs for each value
# 	# ie, for each value, put it in a tuple along with its custom update function
# 	all_agents = [
# 			[(agent['func'], v) for v in agent['values']]
# 			for agent in multi_sim_update_agents
# 		]
# 	# nest the sweeps ... ie cross product of all variations
# 	all_agent_combos = list(itertools.product(*all_agents))

# 	# create new sim_params accordingly
# 	# sweep through all combinations of values, update sim_params, store in list
# 	all_sim_params = []
# 	for agent_combos in all_agent_combos:
# 		new_sim = copy.deepcopy(sim_params)
# 		for agent in agent_combos:
# 			new_sim = agent[0](new_sim, agent[1])
# 		all_sim_params.append(new_sim)
# 	# -

# 	return all_sim_params



# # Main function

def main():

	# ## Custom Utilities

	# +
	mk_time = lambda: dt.datetime.utcnow().isoformat()
	# -

	# ## Results Dir
	# +
	exp_dir = Path('/home/ubuntu/lif_hws_data/results_data/ori_spat_freq_tuning')
	# exp_dir = Path('/Volumes/MagellanSG/PhD/Data/hws_lif')
	# -
	# +
	if not exp_dir.exists():
		exp_dir.mkdir()
	# -


	# # Params

	# ## LGN
	# +
	subset_spat_filts = [
		'berardi84_5a', 'berardi84_5b', 'berardi84_6', 'maffei73_2mid',
		'maffei73_2right', 'so81_2bottom', 'so81_5', 'soodak87_1'
	]
	# -
	# +
	# chief site of variations for experiments
	# templating variables to be used for substitutions:
		# ORI_BIAS_ORIENTATION: average orientation of LGN cells
		# ORI_BIAS_ORIENTATION_CV: Circular variance (ie variation) of len orientation biases
		# SPREAD_RATIO: ratio of spread
		# N_CELLS: number of LGN cells
	#
	lgn_params = do.LGNParams(
		n_cells=30,
		orientation=do.LGNOrientationParams(
			mean_orientation = ArcLength(60, 'deg'),
			circ_var = 0.2
			),
		circ_var=do.LGNCircVarParams('naito_lg_highsf', 'naito'),
		spread=do.LGNLocationParams(
			ratio = 3,
			distribution_alias = 'jin_etal_on',
			orientation = ArcLength(90, 'deg')  # default value too
			),
		filters=do.LGNFilterParams(spat_filters=subset_spat_filts, temp_filters='all'),
		F1_amps=do.LGNF1AmpDistParams()
		)
	# -
	# +
	# max_spat_ext = stimulus.estimate_max_stimulus_spatial_ext_for_lgn(
	# 	spat_res, lgn_params, n_cells=5000, safety_margin_increment=0.1)
	# print(max_spat_ext.mnt)
	# -

	# ## Space Time
	# +
	spat_res=ArcLength(1, 'mnt')
	spat_ext=ArcLength(660, 'mnt')
	"Good high value that should include any/all worst case scenarios"
	temp_res=Time(1, 'ms')
	temp_ext=Time(1000, 'ms')

	st_params = do.SpaceTimeParams(
		spat_ext, spat_res, temp_ext, temp_res,
		array_dtype='float16'
		# array_dtype='float32'
		)
	# -


	# # Simulation

	# ## Meta
	# +
	# Increment on top of what has been made already
	exp_id_prefix = 'HWS'
	new_exp_id = run.mk_incremented_single_exp_dir(exp_dir, exp_id_prefix)

	meta_data = do.SimulationMetaData(
		# 'HWS1',
		exp_id = new_exp_id,
		comments = '''Spatial Freq and Ori tuning without synchrony
		''')
	# -

	# ##  Stim Params
	# +
	spat_freqs = [0.4, 0.8, 1, 1.2, 1.5, 2]
	oris = np.arange(0, 180, 22.5)
	multi_stim_params = do.MultiStimulusGeneratorParams(
		spat_freqs=spat_freqs,
		temp_freqs=[4],
		orientations=oris,
		contrasts=[0.4]
		)
	# -

	# ## LIF Params

	# +
	lif_params = do.LIFParams(
		total_EPSC=3.25
		)
	# -

	# ## Sim Params

	# ### Base
	# +
	sim_params = do.SimulationParams(
		n_simulations=100,
		space_time_params=st_params,
		multi_stim_params=multi_stim_params,
		lgn_params=lgn_params,
		lif_params = lif_params,
		n_trials = 10,
		analytical_convolution=True
		)
	# -
	# +
	synch_params = do.SynchronyParams(
		False,
		jitter=Time(3, 'ms')
		)
	# -

	# ### Multi Sim Params

	# +
	# Define parameter sweeps

	# LGN params can't be swept ... only one lgn layer at a time
	# multi_sim_params_ratios = MultiSimAgents(
	# 	values=[1, 2, 3, 4, 10], attr_path='lgn_params.spread.ratio')

	# multi_sim_params_cv = MultiSimAgents(
	# 	values=[0.1, 0.2, 0.3], attr_path='lgn_params.orientation.circ_var')

	# multi_sim_update_agents = (multi_sim_params_cv, multi_sim_params_ratios)


	# Just make none if want to use only base sim params
	multi_sim_update_agents = None
	# -
	# +
	all_sim_params = mk_all_sim_params(sim_params, multi_sim_update_agents)
	all_sim_params_key_vars = mk_all_sim_params(
		sim_params, multi_sim_update_agents, return_combinations_only=True)
	# -



	# ## Sim Logistics

	# +
	# n_procs = 1
	n_procs = 30  # for large instance (here, 32 cores)
	n_sims_per_partition = 500
	n_partitions, partitioned_n_sims = run.mk_n_simulation_partitions(
		sim_params.n_simulations,
		n_sims_per_partition
		)

	# multi_stim_combos = stimulus.mk_multi_stimulus_params(sim_params.multi_stim_params)
	# -

	# ## prep stimuli
	# +
	# Just params
	multi_stim_combos = stimulus.mk_multi_stimulus_params(sim_params.multi_stim_params)
	multi_stim_combos_key_vars = stimulus.mk_multi_stimulus_params(
		sim_params.multi_stim_params, return_combinations_only=True)
	# Create cache
	# multi_stim_combos = run.create_stimulus(sim_params, force_central_rf_locations=False)
	# -


	# Loop through this
	# unpack sp and stim_p
	# after sim ... get all sim_path_result_paths and bundle with these:
	# Dict[int, Dict['path': Path, 'sim_p': sim_p, 'stim_p': stim_p]]
	# +
	all_param_combos = tuple(
			{'sim_params': sp, 'stim_params': stim_p}
				for sp in all_sim_params
					for stim_p in multi_stim_combos
		)

	# just the variables that are changed
	all_param_combos_key_vars = tuple(
			{
			# sim params may not vary, in which case they will be SimulationParams, so provide None
			'sim_params': (
					None
						if isinstance(sp, do.SimulationParams) else
					sp
				)
			,
			# if a tuple is empty, that means no changing stimuli in multi stimuli
			'stim_params': (
					None
						if len(stim_p) == 0 else
					stim_p
				)
			}
				for sp in all_sim_params_key_vars
					for stim_p in multi_stim_combos_key_vars
		)
	# -

	# ## prep single stim dirs
	# +
	results_dir = exp_dir / meta_data.exp_id
	# -
	# +
	print(f'\n************\n\n{meta_data.exp_id} ... Prepping exp  ...')
	print(f'... with {len(all_param_combos)} sims\n\n')
	print('Making Directories')
	time.sleep(1)
	# can't suppress until stim results not stored by iteration index
	run.prep_results_dir(results_dir, suppress_exc=False)
	run.prep_temp_results_dirs(results_dir=results_dir, n_stims = all_param_combos)
	# run.prep_temp_results_dirs(results_dir=results_dir, n_stims = multi_stim_combos)
	print('made directories')
	# -

	# ## create lgn layers
	# +
	# Create new
	all_lgn_layers = run.create_all_lgn_layers(sim_params, force_central_rf_locations=False)

	# Load old layer
	# -

	# ## Create overlap maps
	# +
	# create lgn layer overlapping regions data
	if synch_params.lgn_has_synchrony:
		all_lgn_overlap_maps = run.create_all_lgn_layer_overlapping_regions(all_lgn_layers, sim_params)
		# the main one, with weights reduced so that after duplication to target LGN cells rates are accurate
		all_adjusted_lgn_overlap_maps = run.mk_adjusted_overlapping_regions_wts(all_lgn_overlap_maps)
	else:
		all_lgn_overlap_maps = None
		all_adjusted_lgn_overlap_maps = None
	# -

	# ## save meta, params, lgn layers
	# +
	# copy the experiment script to the experiment
	this_script = Path(__file__)
	shutil.copy(__file__, results_dir / f'exp_script_{this_script.name}')

	run._save_pickle_file(results_dir / 'meta_data.pkl', meta_data)
	run._save_pickle_file(results_dir / 'simulation_params.pkl', sim_params)
	run._save_pickle_file(results_dir / 'all_simulation_params.pkl', all_param_combos)
	run._save_pickle_file(
		results_dir / 'all_simulation_params_key_vars.pkl',
		all_param_combos_key_vars
		)
	run._save_pickle_file(
		results_dir / 'lgn_layers.pkl',
		cells.mk_contrast_lgn_layer_collection_record(all_lgn_layers)
		)
	run._save_pickle_file(results_dir / 'synch_params.pkl', synch_params )

	if all_lgn_overlap_maps:
		run._save_pickle_file(results_dir / 'lgn_overlap_maps.pkl', all_lgn_overlap_maps)
	if all_adjusted_lgn_overlap_maps:
		run._save_pickle_file(
			results_dir / 'adjusted_lgn_overlap_maps.pkl',
			all_adjusted_lgn_overlap_maps
			)
	# -

	# ## RUN - Parallel Workers or Linear worker

	# +
	print('starting workers')
	time.sleep(1)
	# -

	#####
	# # MultiProc
	####

	pool = mp.Pool(processes=n_procs)

	print(f'Starting Simulations... n_sims: {len(all_param_combos)}, n_procs: {n_procs} ({dt.datetime.utcnow().isoformat()})')


	for i, param_combos in enumerate(all_param_combos):

		pool.apply_async(
			func=run.run_partitioned_single_stim,
			kwds={
				'n_stim': i,
				'params': param_combos['sim_params'],
				'stim_params': param_combos['stim_params'],
				'synch_params': synch_params,
				'lgn_layers': all_lgn_layers,
				'results_dir': results_dir,
				'partitioned_sim_lgn_idxs': partitioned_n_sims,
				# Using ADJUSTED OVERLAP MAP as spikes will be duplicated
				'lgn_overlap_maps': (
						all_adjusted_lgn_overlap_maps
							if synch_params.lgn_has_synchrony else
						None
					),
				'log_print': True,
				'save_membrane_data': False
			}
			)

	pool.close()
	pool.join()

	print(f'{meta_data.exp_id} ... Done simulations ({dt.datetime.utcnow().isoformat()})')
	# -

	# ## Save single_stim_results file index
	# +
	print(f'{meta_data.exp_id} ... Saving result index file')
	result_files_idx = run.get_all_experiment_single_stim_results_files(
			results_dir, all_param_combos
		)
	result_files_params_idx = {
		i: {'path': path, **all_param_combos[i]}
		for i, path in result_files_idx.items()
	}
	run._save_pickle_file(results_dir / 'result_files_params_idx.pkl', result_files_params_idx)
	# -

	print(f'{meta_data.exp_id} ... DONE!\n\n***************\n')


	# collect results from stim_dirs into single results object and save

	# +
	# if want to put all results into single file ... will be pretty big!!
	# maybe best to just aggregate what's necessary per stimulus condition from the pickle files
	# ... in which case, use utility function to get all single-stim files and line up with stim files
	# ... maybe save to pickle file too to ensure line up is accurate?

	# run.save_merge_all_results(
	# 	results_dir,
	# 	multi_stim_combos,
	# 	sim_params
	# 	)
	# -

if __name__ == '__main__':
	main()
