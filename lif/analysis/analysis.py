from os import get_exec_path
from pathlib import Path
from typing import Callable, Literal, List, Tuple, Dict, Union, overload, Optional, Sequence

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd
import plotly.express as px

import lif.lgn.cells as cells
import lif.utils.data_objects as do
import lif.utils.exceptions as exp
import lif.convolution.estimate_real_amp_from_f1 as est_f1
from lif.utils.units.units import ArcLength, Time, TempFrequency
import lif.simulation.run as run
# from ..simulation import run

import lif.receptive_field.filters.filter_functions as ff
import lif.receptive_field.filters.cv_von_mises as cvvm
import lif.stimulus.stimulus as stim

from . import circ_stat



# # Core

# Load data and meta data



def load_sim_params(exp_results_dir: Path) -> do.SimulationParams:
	params_file = exp_results_dir/'simulation_params.pkl'
	params = run._load_pickle_file(params_file)

	return params


def load_synch_params(exp_results_dir: Path) -> do.SynchronyParams:
	params_file = exp_results_dir/'synch_params.pkl'
	params = run._load_pickle_file(params_file)

	return params


def load_key_param_vars(exp_results_dir: Path):
	params_file = exp_results_dir / 'all_simulation_params_key_vars.pkl'
	params = run._load_pickle_file(params_file)

	return params


def load_all_params(
			exp_results_dir: Path
		) -> Tuple[Dict[str, Union[do.SimulationParams, do.GratingStimulusParams]], ...]:
	params_file = exp_results_dir / 'all_simulation_params.pkl'
	params = run._load_pickle_file(params_file)

	return params


def mk_key_stim_vars(exp_results_dir: Path):

	simulation_params = load_sim_params(exp_results_dir)
	stim_key_combinations = stim.mk_multi_stimulus_params(
		simulation_params.multi_stim_params,
		return_combinations_only=True
		)

	return stim_key_combinations


def load_result_params_index(exp_results_dir: Path):
	params_file = exp_results_dir / 'result_files_params_idx.pkl'
	params = run._load_pickle_file(params_file)

	return params


@overload
def load_lgn_layers(
			exp_results_dir: Path,
			contrast: None = None
		) -> do.ContrastLgnLayerCollection: ...
@overload
def load_lgn_layers(
			exp_results_dir: Path,
			contrast: Union[do.ContrastValue, Sequence[float], do.SimulationParams]
		) -> Tuple[do.LGNLayer, ...]: ...
def load_lgn_layers(
			exp_results_dir: Path,
			contrast: Optional[Union[do.ContrastValue, Sequence[float], do.SimulationParams]]=None
		) -> Union[do.ContrastLgnLayerCollection, Tuple[do.LGNLayer, ...]]:

	lgn_layer_records = run.load_lgn_layers(exp_results_dir)
	actual_lgn_layers = cells.mk_contrast_lgn_layer_collection_from_record(lgn_layer_records)

	if contrast:
		contrast_value = (
				contrast
					if not isinstance(contrast, do.SimulationParams) else
				contrast.multi_stim_params.contrasts
			)
		if isinstance(contrast_value, Sequence) and len(contrast_value) > 1:
			raise ValueError(f'Contrast values have more than 1 value ({len(contrast_value)})')
		elif isinstance(contrast_value, Sequence):
			return actual_lgn_layers[do.ContrastValue(contrast_value[0])]
		elif isinstance(contrast_value, do.ContrastValue):
			return actual_lgn_layers[contrast_value]

	return actual_lgn_layers


def get_all_result_files(exp_results_dir: Path) -> Tuple[Path, ...]:

	result_files = tuple(
			file
			for file in exp_results_dir.glob('*.pkl')
			if (run._parse_stim_results_path(file) is not None)
		)

	return result_files


def load_result_file(result_file: Path) -> Tuple[do.SimulationResult, ...]:

	results = run._load_pickle_file(result_file)

	return results




# # Analysis

# For each response
	# get response magnitude (F1, as sine wave amplitude, and max response)
	# get conditions (conditions delta file?  Manual?)
	# aggregate?? Pandas?

# ## PSTHs

@overload
def mk_psth(
			spikes: Union[Time[np.ndarray], do.SimulationResult],
			temp_ext: Time[float],
			bin_width: Time[float] = Time(10, 'ms'),
			use_frequencies: Literal[True] = True
		) -> Tuple[TempFrequency[np.ndarray], Time[np.ndarray]]: ...
@overload
def mk_psth(
			spikes: Union[Time[np.ndarray], do.SimulationResult],
			temp_ext: Time[float],
			bin_width: Time[float] = Time(10, 'ms'),
			use_frequencies: Literal[False] = False
		) -> Tuple[np.ndarray, Time[np.ndarray]]: ...
def mk_psth(
			spikes: Union[Time[np.ndarray], do.SimulationResult],
			temp_ext: Time[float],
			bin_width: Time[float] = Time(10, 'ms'),
			use_frequencies: bool = True
		) -> Tuple[Union[TempFrequency[np.ndarray], np.ndarray], Time[np.ndarray]]:

	bins = np.arange(0, temp_ext.ms+bin_width.ms, bin_width.ms)
	bins = ff.mk_temp_coords(bin_width, Time(temp_ext.base + bin_width.base))

	if isinstance(spikes, Time):
		counts, _ = np.histogram(spikes.ms, bins.ms)

	elif isinstance(spikes, do.SimulationResult):

		all_spikes = Time(np.r_[tuple(
			spikes.get_spikes(n).ms for n in range(spikes.n_trials)
			)]
		, 'ms')

		counts, _ = np.histogram(all_spikes.ms, bins.ms)
		counts = counts / spikes.n_trials

	if use_frequencies:
		counts = TempFrequency(counts / bin_width.s, 'hz')

	return counts, bins


def mk_psth_curve(
			psth_vals: Union[TempFrequency[np.ndarray], np.ndarray],
			sigma: float = 1
		) -> np.ndarray:

	cnts_smooth = gaussian_filter1d(
			(psth_vals if isinstance(psth_vals, np.ndarray) else psth_vals.hz),
			sigma=sigma
		)

	return cnts_smooth


def plot_psth(
			psth_counts: Union[np.ndarray, TempFrequency[np.ndarray]],
			bins: Time[np.ndarray],
			curve: Optional[np.ndarray] = None
		):

	response = (
		psth_counts
			if isinstance(psth_counts, np.ndarray) else
		psth_counts.hz
		)

	plot = (
		px
		.bar(x=bins.ms[:-1], y=response, template='none')
		.update_traces(marker_color='#bbb')
		)

	if curve is not None:
		plot = (
			plot
			.add_trace(
				px.line(x=bins.ms[:-1], y=curve)
				.update_traces(line_color='red')
				.data[0]
				)
			.update_layout(
					xaxis_title='Time (ms)',
					yaxis_title='Resposne (Hz)'
				)

		)
	return plot


# ## Reponse Metrics

# ### F1 fourier

def find_fft_f1(
			psth_vals: np.ndarray, time: Time[np.ndarray],
			temp_freq: TempFrequency[float]
		) -> Optional[float]:

	# with align_freqs = True, bins can be bigger as it will be trimmed
	amps, freqs = est_f1.gen_fft(psth_vals, time, align_freqs=True)

	# check that freqs are properly constructed (ie: 0, 1, 2, 3 ...)
	#  ... and that desired freq is in there
	if not (np.argwhere(freqs == temp_freq.hz)[0][0] == temp_freq.hz):
		# raise ValueError(f'Temporal frequency not in FFT frequencies {temp_freq.hz}(Hz)')
		return None

	# tricky business with using frequency as an index ... but it works if align_freqs has worked
	f1_amp: float = abs(amps[int(temp_freq.hz)])

	return f1_amp

# ### Sin wave fitting

def gen_sin(
	amplitude: float, DC_amp: float,
	time: Time[np.ndarray],
	freq: TempFrequency[float] = TempFrequency(1),
	phase: ArcLength[float] = ArcLength(0)) -> np.ndarray:
	"""Generate Sinusoid of frequency freq over time

	If to be used for deriving real amplitude from empirical F1,
	time must be such that size*temp_res = 1

	Parameters
	----
	time:
		Can be either array or float
	DC_amp:
		Constant amplitude added to whole sinusoid
		Intended to represent the mean firing rate of a neuron

	Returns
	----

	"""

	signal = amplitude * np.sin(freq.hz * 2*np.pi*time.s + phase.rad) + DC_amp  # type: ignore
	# signal: val_gen

	return signal


from scipy.optimize import minimize
from functools import partial

def _opt_amp_dc(
			x: np.ndarray,
			psth_vals: np.ndarray, time: Time[np.ndarray],
			freq: TempFrequency[float],
		):

	# x = (amplitude, DC_amp, phase)
	if x[0] >= 0:
		sin_curve = gen_sin(
				amplitude = x[0],
				DC_amp = x[1],
				phase = ArcLength(x[2], 'rad'),
				freq = freq,
				time = time)

		# rectify
		sin_curve[sin_curve<0] = 0
		value = np.sum(np.abs(psth_vals - sin_curve))  # type: ignore
		return value
	else:
		return 1e10

def optimise_sinusoidal_amp_dc(
			psth_vals: np.ndarray, time: Time[np.ndarray],
			freq: TempFrequency[float],
		):

	# x = (amplitude, DC_amp, phase)
	obj_f = partial(_opt_amp_dc, psth_vals=psth_vals, time=time, freq=freq)

	opt_results = minimize(obj_f, x0=np.array([1, 0, 0]), method='Nelder-Mead')

	return opt_results

def find_sinusoidal_amp(
			psth_vals: np.ndarray, time: Time[np.ndarray],
			freq: TempFrequency[float],
		) -> Optional[float]:

	opt_result = optimise_sinusoidal_amp_dc(psth_vals, time, freq)

	if opt_result.success:
		return opt_result.x[0]
	else:
		return None


# ### Get all Response Metrics



def find_all_response_metrics(
			psth_vals: np.ndarray, time: Time[np.ndarray],
			temp_freq: TempFrequency[float],
		) -> do.V1ResponseMetrics:

	time = Time(time.value[:-1], time.unit)

	all_metrics = do.V1ResponseMetrics(
		max_resp = TempFrequency(psth_vals.max(), 'hz'),
		sin_amp = (
				TempFrequency(val, 'hz')
					if ((val := find_sinusoidal_amp(psth_vals, time, temp_freq)) is not None) else
				None
			),
		fft_f1_amp = (
				TempFrequency(val, 'hz')
					if ((val := find_fft_f1(psth_vals, time, temp_freq)) is not None) else
				None
			)
	)

	return all_metrics


# ### Unifying Function

def analyse_response(
			response: do.SimulationResult,
			space_time_params: do.SpaceTimeParams, stimulus_params: do.GratingStimulusParams,
			bin_width: Time[float] = Time(10, 'ms')
		) -> do.V1ResponseMetrics:

	counts, bins = mk_psth(
		response, space_time_params.temp_ext, bin_width=bin_width)
	psth_curve = mk_psth_curve(counts)

	response_metrics = find_all_response_metrics(psth_curve, bins, stimulus_params.temp_freq)

	return response_metrics


# ## Result File Wrapper

def analyse_result_file(result_file: Path) -> Tuple[do.V1ResponseMetrics, ...]:

	# result object for each layer!
	results = load_result_file(result_file)

	all_response_metrics = list()
	for single_result in results:
		# should have stimulus result_key
		result_params = stim.mk_params_from_stim_signature(single_result.stimulus_results_key)  # type: ignore
		response_metrics = analyse_response(single_result, *result_params)

		all_response_metrics.append(response_metrics)

	return tuple(all_response_metrics)



# ## Experiment Wrapper

def analyse_experiment(
		exp_dir: Path,
		key_params_extraction_func: Callable[
				[do.SimulationParams, do.GratingStimulusParams],
				Dict
			]
		) -> Tuple[
				pd.DataFrame,
				Dict[int, Tuple[Dict, Sequence[do.V1ResponseMetrics]]]
			]:


	result_files = get_all_result_files(exp_dir)
	new_key_params = mk_key_stim_vars(exp_dir)

	try:
		result_params_index = load_result_params_index(exp_dir)
	except ValueError:
		print("Couldn't find/load result_params_index file (probably incomplete simulation)")
		result_params_index = None

	if not (len(result_files) == len(new_key_params)):
		# raise ValueError(
		# 	f'Lengths do not match in {exp_dir}: result files ({len(result_files)}), key params ({len(new_key_params)})')
		print(
			f'WARNING - Lengths do not match in {exp_dir}: result files ({len(result_files)}), key params ({len(new_key_params)})')

	if result_params_index:
		if not (len(result_files) == len(result_params_index)):
			raise ValueError(
				f'Lengths do not match in {exp_dir}: result files ({len(result_files)}), result_params_index ({len(result_params_index)})')

	# all response metrics for each result as a python object

	all_results_response_metrics: Dict[int, Tuple[Dict, Sequence[do.V1ResponseMetrics]]] = dict()

	all_dfs: List[pd.DataFrame] = list()

	# result file
	for result_file in result_files:
		result_index = run._parse_stim_results_path(Path(result_file.name))
		if result_index is None:
			raise ValueError(f'Failed to parse result file ({result_file})')
		if result_params_index:
			if not (result_params_index[result_index]['path'].name == result_file.name):
				raise ValueError(
					f'result file name and params index mismatch: index: {result_index}')

		results = load_result_file(result_file)

		all_response_metrics: Sequence[do.V1ResponseMetrics] = list()
		# hopefully a safe presumption that lgn_layer order is same as result order?!
		# seems baked into the simulation code well enough
		for single_result in results:
			# should have stimulus result_key
			result_params = stim.mk_params_from_stim_signature(single_result.stimulus_results_key)  # type: ignore
			response_metrics = analyse_response(single_result, *result_params)

			all_response_metrics.append(response_metrics)

		# relevant params
		# V-- dictionary (key is param and unit if applicable)
		if result_params_index:
			key_params = key_params_extraction_func(
				result_params_index[result_index]['sim_params'],
				result_params_index[result_index]['stim_params']
				)
		# revert to backup
		else:
			sim_params = load_sim_params(exp_dir)
			if not results[0].stimulus_results_key:
				raise ValueError(
					'results index not available and stimulus key missing, no way to get stimulus params')
			stim_signature = results[0].stimulus_results_key
			st_params, stim_params = stim.mk_params_from_stim_signature(stim_signature)

			key_params = key_params_extraction_func(sim_params, stim_params)


		# double check with the result obje params??

		# put together into a dataframe?
		df = pd.DataFrame(
				data = [
					{
						'LGN Layer idx': i,
						'F1': (resp_met.fft_f1_amp.hz if resp_met.fft_f1_amp else None),
						'Sin Amp': (resp_met.sin_amp.hz if resp_met.sin_amp else None),
						'Max Resp': resp_met.max_resp.hz
					}
					for i, resp_met in enumerate(all_response_metrics)
				]
			)

		for k, val in key_params.items():
			df[k] = val
		df['result_file_name'] = result_file.name
		df['result_file_idx'] = result_index

		# python object
		all_results_response_metrics[result_index] = (key_params, all_response_metrics)
		all_dfs.append(df)

	full_df = pd.concat(all_dfs)

	return full_df, all_results_response_metrics


def mk_results_file_paths(exp_dir: Path) -> Tuple[Path, Path]:

	df_file = exp_dir / 'RESPONSES_DF.parquet'
	resp_metrics_file = exp_dir / 'RESPONSES_OBJ.pkl'

	return df_file, resp_metrics_file


@overload
def load_experiment_analysis(
			exp_dir: Path,
			check_existance: Literal[False] = False,
			remove: Literal[False] = False
		) -> Tuple[
				pd.DataFrame,
				Dict[int, Tuple[Dict, Sequence[do.V1ResponseMetrics]]]
			]: ...
@overload
def load_experiment_analysis(
			exp_dir: Path,
			check_existance: Literal[True],
			remove: bool = False
		) -> bool: ...
def load_experiment_analysis(
			exp_dir: Path,
			check_existance: bool = False,
			remove: bool = False
		) -> Union[
				bool,
				Tuple[
					pd.DataFrame,
					Dict[int, Tuple[Dict, Sequence[do.V1ResponseMetrics]]]
				]
			]:

	df_file, resp_metrics_file = mk_results_file_paths(exp_dir)

	if (not df_file.exists()) or (not resp_metrics_file.exists()):

		if check_existance is True:
			return False

		raise FileExistsError(
			f'Not all results files exist (df: {df_file.exists()}, resp_metrics: {resp_metrics_file.exists()})')

	# they exist
	elif check_existance:
		# remove if exist ... for redoing the analysis
		if remove:
			df_file.unlink()
			resp_metrics_file.unlink()
		return True

	df = pd.read_parquet(df_file)
	resp_metrics = run._load_pickle_file(resp_metrics_file)

	return df, resp_metrics


def save_experiment_analysis(df, resp_metrics, exp_dir):

	if load_experiment_analysis(exp_dir, check_existance=True):
		raise FileExistsError(f'Results file already exist in {exp_dir}')

	df_file, resp_metrics_file = mk_results_file_paths(exp_dir)

	df.to_parquet(df_file)
	run._save_pickle_file(resp_metrics_file, resp_metrics)



# ## Analyse whole run

def get_all_exp_dirs(run_dir: Path) -> Tuple[Path, ...]:

	all_matches = run_dir.glob('HWS*')
	all_exp_dirs = tuple(
			match
				for match in all_matches
			if match.is_dir()
		)

	return all_exp_dirs

def analyse_run(
			run_dir: Path,
			experiment_key_params_func: Callable[
					[do.SimulationParams, do.GratingStimulusParams],
					Dict
				],
			run_key_params_func: Callable[
					[do.SimulationParams, do.SynchronyParams], Dict
				]
		):

	exp_dirs = get_all_exp_dirs(run_dir)

	# check if done already

	all_analysis_already_exists = [
			load_experiment_analysis(exp_dir, check_existance=True)
			for exp_dir in exp_dirs
		]

	if any(all_analysis_already_exists):
		raise exp.SimulationError(
			f'Results files already exist in {sum(all_analysis_already_exists)} experiment dirs')

	# analyse each experiment
	print(f'Analysing all experiments in {run_dir}')
	for i, exp_dir in enumerate(exp_dirs):
		print(f'{i:>4} / {len(exp_dirs)} ... Analysing {exp_dir.name}')
		df, resp_metrics = analyse_experiment(exp_dir, experiment_key_params_func)
		save_experiment_analysis(df, resp_metrics, exp_dir)

	# check all were successful
	all_analysis_successful = [
			load_experiment_analysis(exp_dir, check_existance=True)
			for exp_dir in exp_dirs
		]

	if not all(all_analysis_successful):
		raise exp.SimulationError(
			f'Failed analysis in {sum(all_analysis_successful)} experiments')

	print(f'All experiments analysed and saves successfully')

	# concatenate all DFs

	print('Concatenating all Dataframes')
	all_dfs = []
	for i, exp_dir in enumerate(exp_dirs):

		print(f'{i:>4} / {len(exp_dirs)} ... Analysing {exp_dir.name}')

		df, resp_metrics = load_experiment_analysis(exp_dir)

		sim_params = load_sim_params(exp_dir)
		synch_params = load_synch_params(exp_dir)
		run_key_params = run_key_params_func(sim_params, synch_params)

		for key, val in run_key_params.items():
			df[key] = val

		all_dfs.append(df)

	complete_df = pd.concat(all_dfs)

	complete_df_path = run_dir / "ALL_RESPONSE_DATA.parquet"

	print(f'Concatenated ... saving to {complete_df_path}')

	complete_df.to_parquet(complete_df_path)





# class V1ResponseMetrics(ConversionABC):
#     max_resp: TempFrequency[float]
#     sin_amp: Optional[TempFrequency[float]]
#     fft_f1_amp: Optional[TempFrequency[float]]

		# Save out a basic pickle object too for safe keeping?

		# key_params = new_key_params[result_index]



# # Data files list

# -rw-r--r--  1 errollloyd  staff  41781036 28 Aug 00:48 000.pkl
# -rw-r--r--  1 errollloyd  staff  41788614 28 Aug 00:48 001.pkl
# -rw-r--r--  1 errollloyd  staff  41796103 28 Aug 00:48 002.pkl
# -rw-r--r--  1 errollloyd  staff  41783920 28 Aug 00:48 003.pkl
# -rw-r--r--  1 errollloyd  staff  41793132 28 Aug 00:48 004.pkl
# -rw-r--r--  1 errollloyd  staff  41787044 28 Aug 00:48 005.pkl
# -rw-r--r--  1 errollloyd  staff  41784267 28 Aug 00:48 006.pkl
# -rw-r--r--  1 errollloyd  staff  41784133 28 Aug 00:48 007.pkl
# -rw-r--r--  1 errollloyd  staff  38267681 28 Aug 00:48 008.pkl
# -rw-r--r--  1 errollloyd  staff  38278165 28 Aug 00:48 009.pkl
# -rw-r--r--  1 errollloyd  staff  38298201 28 Aug 00:48 010.pkl
# -rw-r--r--  1 errollloyd  staff  38318761 28 Aug 00:48 011.pkl
# -rw-r--r--  1 errollloyd  staff  38318844 28 Aug 00:48 012.pkl
# -rw-r--r--  1 errollloyd  staff  38320293 28 Aug 00:48 013.pkl
# -rw-r--r--  1 errollloyd  staff  38308585 28 Aug 00:48 014.pkl
# -rw-r--r--  1 errollloyd  staff  38289094 28 Aug 00:48 015.pkl
# -rw-r--r--  1 errollloyd  staff  35884718 28 Aug 00:48 016.pkl
# -rw-r--r--  1 errollloyd  staff  35890920 28 Aug 00:48 017.pkl
# -rw-r--r--  1 errollloyd  staff  35896356 28 Aug 00:48 018.pkl
# -rw-r--r--  1 errollloyd  staff  35911992 28 Aug 00:48 019.pkl
# -rw-r--r--  1 errollloyd  staff  35923684 28 Aug 00:48 020.pkl
# -rw-r--r--  1 errollloyd  staff  35917616 28 Aug 00:48 021.pkl
# -rw-r--r--  1 errollloyd  staff  35907147 28 Aug 00:48 022.pkl
# -rw-r--r--  1 errollloyd  staff  35898186 28 Aug 00:48 023.pkl
# -rw-r--r--  1 errollloyd  staff  33867754 28 Aug 00:48 024.pkl
# -rw-r--r--  1 errollloyd  staff  33866496 28 Aug 00:48 025.pkl
# -rw-r--r--  1 errollloyd  staff  33865053 28 Aug 00:48 026.pkl
# -rw-r--r--  1 errollloyd  staff  33842615 28 Aug 00:48 027.pkl
# -rw-r--r--  1 errollloyd  staff  33839522 28 Aug 00:49 028.pkl
# -rw-r--r--  1 errollloyd  staff  33848723 28 Aug 00:49 029.pkl
# -rw-r--r--  1 errollloyd  staff  33848009 28 Aug 00:49 030.pkl
# -rw-r--r--  1 errollloyd  staff  33855827 28 Aug 00:49 031.pkl
# -rw-r--r--  1 errollloyd  staff  31624516 28 Aug 00:49 032.pkl
# -rw-r--r--  1 errollloyd  staff  31625418 28 Aug 00:49 033.pkl
# -rw-r--r--  1 errollloyd  staff  31612285 28 Aug 00:49 034.pkl
# -rw-r--r--  1 errollloyd  staff  31594552 28 Aug 00:49 035.pkl
# -rw-r--r--  1 errollloyd  staff  31579206 28 Aug 00:49 036.pkl
# -rw-r--r--  1 errollloyd  staff  31589542 28 Aug 00:49 037.pkl
# -rw-r--r--  1 errollloyd  staff  31606584 28 Aug 00:49 038.pkl
# -rw-r--r--  1 errollloyd  staff  31628570 28 Aug 00:49 039.pkl
# -rw-r--r--  1 errollloyd  staff  29858681 28 Aug 00:49 040.pkl
# -rw-r--r--  1 errollloyd  staff  29857681 28 Aug 00:49 041.pkl
# -rw-r--r--  1 errollloyd  staff  29852516 28 Aug 00:49 042.pkl
# -rw-r--r--  1 errollloyd  staff  29851398 28 Aug 00:49 043.pkl
# -rw-r--r--  1 errollloyd  staff  29842431 28 Aug 00:49 044.pkl
# -rw-r--r--  1 errollloyd  staff  29849144 28 Aug 00:49 045.pkl
# -rw-r--r--  1 errollloyd  staff  29858816 28 Aug 00:49 046.pkl
# -rw-r--r--  1 errollloyd  staff  29855271 28 Aug 00:49 047.pkl
# -rw-r--r--  1 errollloyd  staff      7501 28 Aug 00:47 all_simulation_params.pkl
# -rw-r--r--  1 errollloyd  staff       178 28 Aug 00:47 all_simulation_params_key_vars.pkl
# -rw-r--r--  1 errollloyd  staff     14286 28 Aug 00:47 exp_script_exp_base_ORI_BIAS_ORIENTATION_CV_0.4_SPREAD_RATIO_2_ORI_BIAS_ORIENTATION_45_N_CELLS_30.py
# -rw-r--r--  1 errollloyd  staff   1792269 28 Aug 00:47 lgn_layers.pkl
# -rw-r--r--  1 errollloyd  staff       206 28 Aug 00:47 meta_data.pkl
# -rw-r--r--  1 errollloyd  staff      9308 28 Aug 00:49 result_files_params_idx.pkl
# -rw-r--r--  1 errollloyd  staff      1701 28 Aug 00:47 simulation_params.pkl
# -rw-r--r--  1 errollloyd  staff       158 28 Aug 00:47 synch_params.pkl


