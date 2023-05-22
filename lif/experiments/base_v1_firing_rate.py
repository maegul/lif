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
	orientation=do.LGNOrientationParams(ArcLength(30), circ_var=1),
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
	array_dtype='float32'
	)
# -
# +
multi_stim_params = do.MultiStimulusGeneratorParams(
	spat_freqs=[0.8], temp_freqs=[4], orientations=[90], contrasts=[0.4]
	)
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
test_dir = Path('/home/ubuntu/lif_hws/work/results_data')
# -
# +
for total_epsc in np.linspace(3, 5, 5):
	print(f'Total EPSC = {total_epsc}')
	sim_params.lif_params.total_EPSC = total_epsc
	results = run.run_simulation(sim_params)
	run.save_simulation_results(
			results_dir = test_dir,
			sim_results = results,
			comments = 'test run'
		)
# -
