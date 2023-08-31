from pathlib import Path

import lif.analysis.analysis as anlys
import lif.utils.data_objects as do
import lif.simulation.run as run

from lif.utils.units.units import ArcLength, Time, TempFrequency

import multiprocessing as mp


base_dir = Path('~/lif_hws_data/results_data').expanduser()

ori_spat_freq_run = base_dir / 'ori_spat_freq_tuning'

if not ori_spat_freq_run.exists():
	raise ValueError(f'run dir does not exist: {ori_spat_freq_run}')

run_dir = ori_spat_freq_run

def spat_freq_ori_key_params_extract(sim: do.SimulationParams, stim: do.GratingStimulusParams):

	return {
		'Orientation (deg)': stim.orientation.deg,
		'Spat Freq (cpd)': stim.spat_freq.cpd
	}

def hws_conditions_extract(sim: do.SimulationParams, synch: do.SynchronyParams):

	return {
		"Spread Ratio": sim.lgn_params.spread.ratio,
		"Spread Ori (deg)": sim.lgn_params.spread.orientation.deg,
		"Orientation Bias (deg)": sim.lgn_params.orientation.mean_orientation.deg,
		"Orientation Bias Circ Var": sim.lgn_params.orientation.circ_var,
		"Synchronous LGN": synch.lgn_has_synchrony,
		"Synchrony Jitter (ms)": synch.jitter.ms

	}


def run_experiment_analysis(i, exp_dir, experiment_key_params_func):
	print(f'Running analysis on experiment {i} in {exp_dir}')
	df, resp_metrics = anlys.analyse_experiment(exp_dir, experiment_key_params_func)
	print(f'Saving analysis for experiment {i}')
	anlys.save_experiment_analysis(df, resp_metrics, exp_dir)


n_procs = 3

def main():

	pool = mp.Pool(processes=n_procs)

	exp_dirs = anlys.get_all_exp_dirs(run_dir)

	for i, exp_dir in enumerate(exp_dirs):

		pool.apply_async(
			func=run_experiment_analysis,
			kwds={
				'i': i,
				'exp_dir': exp_dir,
				'experiment_key_params_func': spat_freq_ori_key_params_extract,
			}
			)

	pool.close()
	pool.join()


if __name__ == '__main__':
	main()
