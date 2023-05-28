# # Imports
# +
from pathlib import Path
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

# ## Setting up params

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
# +
lif_params = do.LIFParams(
	total_EPSC=3
	)
# -
# +
sim_params = do.SimulationParams(
	n_simulations=100,
	space_time_params=st_params,
	multi_stim_params=multi_stim_params,
	lgn_params=lgn_params,
	lif_params = lif_params,
	n_trials = 10
	)
# -


# # Simulation
# +
import sys
# -
# +
mk_time = lambda: dt.datetime.utcnow().isoformat()
# -
# +
test_dir = Path('/home/ubuntu/lif_hws/work/results_data')
# -

# ## Parallel Params
# +
all_stim_params = [0, 0.2, 0.4, 0.8,  1, 1.2, 1.6, 2, 4, 8]
# -
# +
# distribute the params
n_procs = 3

parallel_params = {
	str(i+1): list()
	for i in range(n_procs)
}
for i, e in enumerate(all_stim_params):
	parallel_params[str((i%n_procs)+1)].append(e)
# -
# +
# predefined params
# parallel_params = {
# 	'1': [0, 0.2, 0.4,],
# 	'2': [0.8,  1, 1.2],
# 	'3': [1.6, 2, 4]
# }
# -
# +
parallel_param_arg = sys.argv[1]
# -

# ### Arg 0 ... prep stimuli in single thread
# +
# If "0" ... then prepare all stimuli ... presumes that variable parameters are stimuli
if parallel_param_arg == '0':

	multi_stim_params = do.MultiStimulusGeneratorParams(
		spat_freqs=all_stim_params, temp_freqs=[4], orientations=[90], contrasts=[0.4]
		)
	multi_params = stimulus.mk_multi_stimulus_params(multi_stim_params)
	stimulus.mk_stimulus_cache(st_params, multi_params)

	sys.exit()
# -
# +
current_params = parallel_params[parallel_param_arg]
# -
# +
lif_params = do.LIFParams(
	total_EPSC=3
	)
multi_stim_params = do.MultiStimulusGeneratorParams(
	spat_freqs=current_params,
	temp_freqs=[4], orientations=[90], contrasts=[0.4]
	)
# -
# +
sim_params = do.SimulationParams(
	n_simulations=1000,
	space_time_params=st_params,
	multi_stim_params=multi_stim_params,
	lgn_params=lgn_params,
	lif_params = lif_params,
	n_trials = 10
	)
# -

# ## Run!!

# +
print(mk_time())
print('Running simulation')
if 'parallel_params' in locals():
	print(f'Parallel Param: {parallel_param_arg}, current: {current_params}')

results = run.run_simulation(sim_params)

print(mk_time())
print('Saving results')
run.save_simulation_results(
	results_dir = test_dir,
	sim_results = results,
	comments = 'HWS2: spatial frequency tuning of untuned inputs with ratio1, to get a base line'
	)
# -
