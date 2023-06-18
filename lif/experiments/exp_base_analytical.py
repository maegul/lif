# # Imports
# +
import time
from pathlib import Path
from typing import Tuple, Union
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


# # Params
# +
subset_spat_filts = [
	'berardi84_5a', 'berardi84_5b', 'berardi84_6', 'maffei73_2mid',
	'maffei73_2right', 'so81_2bottom', 'so81_5', 'soodak87_1'
]
# -
# +
lgn_params = do.LGNParams(
	n_cells=30,
	orientation=do.LGNOrientationParams(ArcLength(30), circ_var=0),
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
# +
mk_time = lambda: dt.datetime.utcnow().isoformat()
# -
# +
exp_dir = Path('/Volumes/MagellanSG/PhD/Data/hws_lif')
# exp_dir = Path('/home/ubuntu/lif_hws/work/results_data')
# -

# ## Parallel Params
# +
meta_data = do.SimulationMetaData('HWS0', 'Test run with spatial frequency tuning')

all_stim_params = [0, 0.2, 0.4, 0.8,  1, 1.2, 1.6, 2, 4]
multi_stim_params = do.MultiStimulusGeneratorParams(
	spat_freqs=all_stim_params,
	temp_freqs=[4],
	orientations=[90],
	contrasts=[0.4]
	)
# -
# +
lif_params = do.LIFParams(
	total_EPSC=3
	)
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
# +
# n_procs = 2
n_procs = 3
n_sims_per_partition = 500
n_partitions, partitioned_n_sims = run.mk_n_simulation_partitions(
	sim_params.n_simulations,
	n_sims_per_partition
	)

# multi_stim_combos = stimulus.mk_multi_stimulus_params(sim_params.multi_stim_params)
# -

# prep stimuli
# +
multi_stim_combos = stimulus.mk_multi_stimulus_params(sim_params.multi_stim_params)
# multi_stim_combos = run.create_stimulus(sim_params, force_central_rf_locations=False)
# -

# prep single stim dirs
# +
results_dir = exp_dir / meta_data.exp_id
# -
# +
print('Making Directories')
time.sleep(1)
run.prep_results_dir(results_dir)
run.prep_temp_results_dirs(results_dir=results_dir, n_stims = multi_stim_combos)
print('made directories')
# -

# create lgn layers
# +
all_lgn_layers = run.create_all_lgn_layers(sim_params, force_central_rf_locations=False)
# -

# save meta, params, lgn layers
# +
run._save_pickle_file(results_dir / 'meta_data.pkl', meta_data)
run._save_pickle_file(results_dir / 'simulation_params.pkl', sim_params)
run._save_pickle_file(
	results_dir / 'lgn_layers.pkl',
	cells.mk_contrast_lgn_layer_collection_record(all_lgn_layers)
	)
# -

# pool sims into n_procs multi proc

# +
print('starting workers')
time.sleep(1)
# -
# +
# print(f'Starting Simulations... n_sims: {len(multi_stim_combos)}, n_procs: {n_procs} ({dt.datetime.utcnow().isoformat()})')
# for i, stim_param in enumerate(multi_stim_combos):

# 	kwds={
# 		'params': sim_params,
# 		'n_stim': i,
# 		'stim_params': stim_param,
# 		'lgn_layers': all_lgn_layers,
# 		'results_dir': results_dir,
# 		'partitioned_sim_lgn_idxs': partitioned_n_sims,
# 		'log_print': True
# 	}
# 	run.run_partitioned_single_stim(**kwds)


# print(f'Done simulations ({dt.datetime.utcnow().isoformat()})')

pool = mp.Pool(processes=n_procs)

print(f'Starting Simulations... n_sims: {len(multi_stim_combos)}, n_procs: {n_procs} ({dt.datetime.utcnow().isoformat()})')
for i, stim_param in enumerate(multi_stim_combos):

	pool.apply_async(
		func=run.run_partitioned_single_stim,
		kwds={
			'params': sim_params,
			'n_stim': i,
			'stim_params': stim_param,
			'lgn_layers': all_lgn_layers,
			'results_dir': results_dir,
			'partitioned_sim_lgn_idxs': partitioned_n_sims,
			'log_print': True
		}
		)

pool.close()
pool.join()

print(f'Done simulations ({dt.datetime.utcnow().isoformat()})')
# -

# +

# -

# Save single_stim_results file index
# +
result_files_idx = run.get_all_experiment_single_stim_results_files(
		results_dir, multi_stim_combos
	)
run._save_pickle_file(results_dir / 'result_files_idx.pkl', result_files_idx)
# -

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

