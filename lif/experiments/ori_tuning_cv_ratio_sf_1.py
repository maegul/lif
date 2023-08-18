# # Imports
# +
import shutil
import itertools
import copy
import time
from pathlib import Path
from typing import Tuple, Union, Dict, List, TypedDict, Callable, Optional
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


class MultiSimAgents(TypedDict):
	values: List
	func: Callable


def mk_all_sim_params(
		sim_params: do.SimulationParams,
		multi_sim_update_agents: Optional[Tuple[MultiSimAgents, ...]]
		) -> List[do.SimulationParams]:
	# +

	if multi_sim_update_agents is None:
		return [sim_params]

	# repeat funcs for each value
	all_agents = [
			[(agent['func'], v) for v in agent['values']]
			for agent in multi_sim_update_agents
		]
	# nest the sweeps
	all_agent_combos = list(itertools.product(*all_agents))

	# create sim_params accordingly
	all_sim_params = []
	for agent_combos in all_agent_combos:
		new_sim = copy.deepcopy(sim_params)
		for agent in agent_combos:
			new_sim = agent[0](new_sim, agent[1])
		all_sim_params.append(new_sim)
	# -

	return all_sim_params

def main():

	# ## Custom Utilities

	# +
	mk_time = lambda: dt.datetime.utcnow().isoformat()
	# -

	# ## Results Dir
	# +
	exp_dir = Path('/home/ubuntu/lif_hws_data/results_data')
	# exp_dir = Path('/Volumes/MagellanSG/PhD/Data/hws_lif')
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
	lgn_params = do.LGNParams(
		n_cells=30,
		orientation=do.LGNOrientationParams(ArcLength(0), circ_var=0),
		circ_var=do.LGNCircVarParams('naito_lg_highsf', 'naito'),
		spread=do.LGNLocationParams(ratio=1, distribution_alias='jin_etal_on'),
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
	meta_data = do.SimulationMetaData(
		'HWS7',
		'''Orientation Tuning with increasingly focused Sagar orientation and specific ratio...
		... but at SF 1
		Orientations x (circ-vars x ratios) at spat_freq = 1
		''')
	# -

	# ##  Stim Params
	# +
	# spat_freqs = [0, 0.2, 0.4, 0.8,  1, 1.2, 1.6, 2, 4]
	# spat_freqs = [0.4, 0.8,  1, 1.2, 1.6, 2, 4]
	orientations = list(np.arange(0, 180, 22.5))

	multi_stim_params = do.MultiStimulusGeneratorParams(
		spat_freqs=[1],  # incrementing this up
		temp_freqs=[4],
		orientations=orientations,
		contrasts=[0.4]
		)
	# -

	# ## LIF Params

	# +
	lif_params = do.LIFParams(
		total_EPSC=3
		)
	# -

	# ## Sim Params

	# ### Base
	# +
	sim_params = do.SimulationParams(
		n_simulations=1000,
		space_time_params=st_params,
		multi_stim_params=multi_stim_params,
		lgn_params=lgn_params,
		lif_params = lif_params,
		n_trials = 10,
		analytical_convolution=True
		)
	# -

	# ### Multi Sim Params

	# +
	# Define parameter sweeps
	multi_sim_params_ratios = [1, 2, 3, 4, 5, 6]
	def sim_param_update_ratio(sim_params: do.SimulationParams, value):
		new_sim_params = copy.deepcopy(sim_params)
		new_sim_params.lgn_params.spread.ratio = value
		return new_sim_params

	multi_sim_params_cv = [0, 0.2, 0.4, 0.6, 0.8, 1]
	def sim_param_update_cv(sim_params: do.SimulationParams, value):
		new_sim_params = copy.deepcopy(sim_params)
		new_sim_params.lgn_params.orientation.circ_var = value
		return new_sim_params

	multi_sim_update_agents: Tuple[MultiSimAgents, ...] = (
		{'values': multi_sim_params_ratios, 'func': sim_param_update_ratio },
		{'values': multi_sim_params_cv, 'func': sim_param_update_cv },
		)

	# Just make none if want to use only base sim params
	# multi_sim_update_agents = None
	# -
	# +
	all_sim_params = mk_all_sim_params(sim_params, multi_sim_update_agents)
	# -


	# ## Sim Logistics

	# +
	# n_procs = 1
	n_procs = 3
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

	# ## save meta, params, lgn layers
	# +
	# copy the experiment script to the experiment
	this_script = Path(__file__)
	shutil.copy(__file__, results_dir / f'exp_script_{this_script.name}')

	run._save_pickle_file(results_dir / 'meta_data.pkl', meta_data)
	run._save_pickle_file(results_dir / 'simulation_params.pkl', sim_params)
	run._save_pickle_file(results_dir / 'all_simulation_params.pkl', all_param_combos)
	run._save_pickle_file(
		results_dir / 'lgn_layers.pkl',
		cells.mk_contrast_lgn_layer_collection_record(all_lgn_layers)
		)
	# -

	# ## RUN - Parallel Workers or Linear worker

	# +
	print('starting workers')
	time.sleep(1)
	# -
	# +
	# print(f'Starting Simulations... n_sims: {len(all_param_combos)}, n_procs: {n_procs} ({dt.datetime.utcnow().isoformat()})')


	# # for i, stim_param in enumerate(multi_stim_combos):
	# for i, param_combos in enumerate(all_param_combos):

	# 	kwds={
	# 		'n_stim': i,
	# 		'params': param_combos['sim_params'],
	# 		'stim_params': param_combos['stim_params'],
	# 		'lgn_layers': all_lgn_layers,
	# 		'results_dir': results_dir,
	# 		'partitioned_sim_lgn_idxs': partitioned_n_sims,
	# 		'log_print': True,
	# 		'save_membrane_data': False
	# 	}
	# 	run.run_partitioned_single_stim(**kwds)


	# print(f'Done simulations ({dt.datetime.utcnow().isoformat()})')

	#####
	# MultiProc
	####

	pool = mp.Pool(processes=n_procs)

	print(f'Starting Simulations... n_sims: {len(multi_stim_combos)}, n_procs: {n_procs} ({dt.datetime.utcnow().isoformat()})')


	for i, param_combos in enumerate(all_param_combos):

		pool.apply_async(
			func=run.run_partitioned_single_stim,
			kwds={
				'n_stim': i,
				'params': param_combos['sim_params'],
				'stim_params': param_combos['stim_params'],
				'lgn_layers': all_lgn_layers,
				'results_dir': results_dir,
				'partitioned_sim_lgn_idxs': partitioned_n_sims,
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