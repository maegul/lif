# # Imports

# +
from pathlib import Path

# import lif.analysis.analysis as anlys
import lif.analysis.analysis as anlys
import lif.utils.data_objects as do
import lif.simulation.run as run

from lif.utils.units.units import ArcLength, Time, TempFrequency

# -


# # Load Data

# +
# big experimental run
base_dir = Path('/Volumes/MagellanSG/PhD/Data/hws_lif/analysis_proto')
# -
# +
exp_dir = base_dir / 'HWS0117'
# -

# ## Load Params

# +
# all key var params
key_var_params = anlys.load_key_param_vars(exp_dir)
# -
# +
type(key_var_params[0])
# -
# +
key_var_params
# -
# +
all_params = anlys.load_all_params(exp_dir)
# -
type(all_params[10])
len(all_params)
all_params[10]

# +
key_params = [
	(p['stim_params'].orientation, p['stim_params'].spat_freq)
	for p in all_params
]
# -

# +
anlys.stim.mk_multi_stimulus_params(
		all_params[10]['sim_params'].multi_stim_params, return_combinations_only=True)
# -
# +
new_key_params = anlys.mk_key_stim_vars(exp_dir)
new_key_params[0]
# -

# +
result_params_index = anlys.load_result_params_index(exp_dir)
# -
# +
result_params_index[0].keys()
result_params_index[0]['stim_params']
result_params_index[0]['path']
# -


# checking best way to get params
# +
test = list(exp_dir.glob('*.pkl'))

for t in test:
	print(t.name, run._parse_stim_results_path(t))
# -
# +
result_files = sorted(anlys.get_all_result_files(exp_dir))
# -
# +
sorted(result_files)
any('000.pkl' == rf.name for rf in result_files)
any('000.pkl' == rf.name for rf in test)
# -
# +
for rf in result_files:
	print(rf.name, result_params_index[run._parse_stim_results_path(Path(rf.name))]['path'].name)
# -
# +
for i, rpi in result_params_index.items():
	print(i, rpi['path'].name)
# -
# +
for (i, rpi), result_file in zip(result_params_index.items(), sorted(result_files)):
	print(i, rpi['path'].name, result_file.name)
# -
# +
for i in sorted(result_params_index):
	rpi = result_params_index[i]
	result_file = result_files[i]

	result_obj = anlys.load_result_file(result_file)
	result_params = anlys.stim.mk_params_from_stim_signature(result_obj[0].stimulus_results_key)  # type: ignore
	print(
		i, '\n',
		(result_params[1].spat_freq == rpi['stim_params'].spat_freq),
		result_params[1].spat_freq, rpi['stim_params'].spat_freq,
		'\n',
		(result_params[1].orientation == rpi['stim_params'].orientation),
		result_params[1].orientation, rpi['stim_params'].orientation,
		'\n'
		)
	print('')
# -
# +
run._parse_stim_results_path(Path('000.pkl'))
# -
result_params_index[0]['path'].name

# +
result_files = anlys.get_all_result_files(exp_dir)
# -
# +
sorted(result_files)
# -
# +
# result object for each layer!
test = anlys.load_result_file(result_files[20])
len(test)
# result of a single layer
single_result = test[10]
# -

# +
single_result.stimulus_results_key
# -
# +
result_params = anlys.stim.mk_params_from_stim_signature(single_result.stimulus_results_key)
# -
# +
result_params[1].spat_freq, result_params[1].orientation
# -

# +
stim_params = result_params[1]
t = new_key_params[20]

for kp in t:
	print(getattr(stim_params, kp[1]))
# -

# +
lgn_layers = run.load_lgn_layers(exp_dir)
# -
type(lgn_layers)
lgn_layers.keys()
lgn_layer = lgn_layers[result_params[1].contrast]

len(lgn_layer)
type(lgn_layer[0])


# +
sim_params = anlys.load_sim_params(exp_dir)
# -
# +
lgn_layers = anlys.load_lgn_layers(exp_dir, sim_params)
test = lgn_layers[10]
test.rf_distance_scale
# -


# +
single_result.check_n_trials_consistency()
# -
# +
single_result.get_spikes(0)
# -


# # Basic Analysis
# max, fit sin wave at temp_freq and get amplitude (and DC)? -> response

# ## Load single result object

# +
result_files = anlys.get_all_result_files(exp_dir)
# -
# +
i=12
new_key_params[i]
# ((0.8, 'spat_freq'), (90.0, 'orientation'))
# -
# +
# result object for each layer!
test = anlys.load_result_file(result_files[i])
len(test)
# result of a single layer
single_result = test[10]
# -

# ## PSTH of V1 spikes

# +
trial_n = 3
# -
single_result.n_trials

# +
v1_spikes = single_result.get_spikes(trial_n)
# -
# +
# counts, bins = anal.mk_psth(v1_spikes, sim_params.space_time_params.temp_ext)
counts, bins = anlys.mk_psth(
	single_result, sim_params.space_time_params.temp_ext, bin_width=Time(10, 'ms'))
# -
# +
anlys.plot_psth(counts, bins).show()
# -
# +
psth_curve = anlys.mk_psth_curve(counts)
# -
# +
psth_curve = anlys.mk_psth_curve(counts, sigma=1)
anlys.plot_psth(counts, bins, psth_curve).show()
# -


# ## Get response metrics

# ### FFT?

# +
# raises valueerror if can't aligned freqs to temporal resolution
# presumes temp_ext is 1 second and works only if temp_freq is an integer
amps, freqs = anlys.est_f1.gen_fft(psth_curve, bins, align_freqs=True)
# -
# +
anlys.px.line(x=freqs, y=abs(amps)).show()
# -
# +
# tricky business with using frequency as an index ... but it works if align_freqs has worked
abs(amps[int(result_params[1].temp_freq.hz)])
# -

import numpy as np
# +
temp_freq = result_params[1].temp_freq
np.any(freqs == temp_freq.hz)

np.argwhere(freqs == temp_freq.hz)[0][0] == temp_freq.hz
# -


# ### Fitting a sin wave

# +
import numpy as np
from lif.utils.units.units import ArcLength
# -
# +
y = anlys.gen_sin(
	time=bins,
	freq=TempFrequency(4),
	amplitude=30,
	DC_amp=10,
	phase=ArcLength(0, 'rad'))
anlys.px.line(x=bins.s, y=y).show()
# -
# +
y = anlys.gen_sin(time=bins,
	freq=TempFrequency(4),
	amplitude=30,
	DC_amp=10,
	phase=ArcLength(np.pi/2, 'rad'))
anlys.px.line(x=bins.s, y=y).show()
# -

# +
temp_freq = result_params[1].temp_freq
bins_time = Time(bins.value[:-1], bins.unit)
sin_opt = anlys.optimise_sinusoidal_amp_dc(psth_curve, bins_time, temp_freq)
sin_opt
# -

# +
ops = sin_opt.x

opt_sin_curve = anlys.gen_sin(
		amplitude=ops[0], DC_amp=ops[1], time=bins, freq=temp_freq, phase=ArcLength(ops[2], 'rad')
	)
# opt_sin_curve[opt_sin_curve<0] = 0
# -
# +
fig = (
	anlys.px.line(x=bins.s, y=opt_sin_curve)
	.add_scatter(
		mode='lines',
		x=bins.s[:-1], y=psth_curve,
		line_color = 'red'
		)

	)
fig.show()
# anal.px.line(x=bins.s[:-1], y=psth_curve).show()
# -

# +
anlys.find_sinusoidal_amp(psth_curve, bins_time, temp_freq)
# -



# ### Generic function

# +
anlys.find_all_response_metrics(psth_curve, bins, temp_freq)
# -
result_params





# +
# big experimental run
base_dir = Path('/Volumes/MagellanSG/PhD/Data/hws_lif/analysis_proto')
# -
# +
exp_dir = base_dir / 'HWS0117'
# -

# +
result_files = anlys.get_all_result_files(exp_dir)
# -
# +
# result object for each layer!
test = anlys.load_result_file(result_files[20])
len(test)
# result of a single layer
single_result = test[10]
# -
# +
result_params = anlys.stim.mk_params_from_stim_signature(single_result.stimulus_results_key)
# -
# +
response_metrics = anlys.analyse_response(single_result, *result_params)
# -

anlys.px.line(psth_curve).show()



# ### Whole Experiment Functions

# +
def spat_freq_ori_key_params_extract(sim: do.SimulationParams, stim: do.GratingStimulusParams):

	return {
		'Orientation (deg)': stim.orientation.deg,
		'Spat Freq (cpd)': stim.spat_freq.cpd
	}
# -
# +
df, resp_metrics = anlys.analyse_experiment(exp_dir, spat_freq_ori_key_params_extract)
# -
# +
df.to_parquet(exp_dir / 'RESPONSES_DF.parquet')
anlys.run._save_pickle_file(exp_dir / 'RESPONSES_OBJ.pkl', resp_metrics)
# -
# +
anlys.save_experiment_analysis(df, resp_metrics, exp_dir)
# -
# +
anlys.load_experiment_analysis(exp_dir, check_existance=True)
df, resp = anlys.load_experiment_analysis(exp_dir)
# -
# +
df['F1'].isna().sum()
# -
# +
df[~df['Sin Amp'].isna()]['F1'].describe()
df[df['Sin Amp'].isna()]['F1'].describe()
# -
# +
anlys.px.scatter(df, x='F1', y='Sin Amp').show()
anlys.px.scatter(df, x='F1', y='Max Resp').show()
# -

# +
# res_idx, lgn_n = (1, 87)
# res_idx, lgn_n = (29, 59)
res_idx, lgn_n = (4, 0)
# -
# +
results = anlys.load_result_file(exp_dir/f'{res_idx:0>3}.pkl')
# -
# +
result = results[lgn_n]
# -
# +
cnts, bins = anlys.mk_psth(result, Time(1, 's'))
# -
# +
anlys.plot_psth(cnts, bins).show()
# -




# +
df.to_parquet(exp_dir / 'RESULTS_DF.parquet')
# -
# +
import pandas as pd
from pandas import testing
# -
# +
new_df = pd.read_parquet(exp_dir / 'RESULTS_DF.parquet')
# -
# +
(new_df == df).sum()
new_df.isna().sum()
# -
df[df['Sin Amp'].isna()]
df[df['F1'].isna()]['Max Resp'].describe()
# +
df.loc[(new_df['F1'] != df['F1'])]
new_df.loc[(new_df['F1'] != df['F1'])]
# -
# +
np.all(new_df == df)
# -
# +
new_df[]
# -
# +
testing.assert_frame_equal(df,  new_df)
# -

sim_params = anlys.load_sim_params(exp_dir)



# +
def spat_freq_ori_key_params_extract(sim: do.SimulationParams, stim: do.GratingStimulusParams):

	return {
		'Orientation (deg)': stim.orientation.deg,
		'Spat Freq (cpd)': stim.spat_freq.cpd
	}

def hws_conditions_extract(sim: do.SimulationParams, synch: do.SynchronyParams):

	return {
		"Spread Ratio": sim_params.lgn_params.spread.ratio,
		"Spread Ori (deg)": sim_params.lgn_params.spread.orientation.deg,
		"Orientation Bias (deg)": sim_params.lgn_params.orientation.mean_orientation.deg,
		"Orientation Bias Circ Var": sim_params.lgn_params.orientation.circ_var,
		"Synchronous LGN": synch.lgn_has_synchrony,
		"Synchrony Jitter (ms)": synch.jitter.ms

	}
# -
# +
# -
# +
anlys.analyse_run(base_dir, spat_freq_ori_key_params_extract, hws_conditions_extract)
# -
# +
exp_dirs = anlys.get_all_exp_dirs(base_dir)
any(
	anlys.load_experiment_analysis(ed, check_existance=True)
	for ed in exp_dirs
	)
# -
# +
for ed in exp_dirs:
	anlys.load_experiment_analysis(ed, check_existance=True, remove=True)
# -

# Then ... organise!
#  - spat_freq + ori conditions (see key params): response ... for each simulation (dataframe?)
# Then ... experiment results
#  - get simulation results for each simulation
#  - get conditions for each simulation
#    - what differs them from each other?  Manually define params then extract from params objs
# - combine into bigger dataframe






















