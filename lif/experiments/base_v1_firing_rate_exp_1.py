from .base_v1_firing_params import *

# # Simulation
# +
mk_time = lambda: dt.datetime.utcnow().isoformat()
# -
# +
test_dir = Path('/home/ubuntu/lif_hws/work/results_data')
# -
# +
for total_epsc in [3, 3.5]:
	print(mk_time())
	print('Running simulation')

	print(f'Total EPSC = {total_epsc}')
	sim_params.lif_params.total_EPSC = total_epsc

	results = run.run_simulation(sim_params)

	print(mk_time())
	print('Saving results')
	run.save_simulation_results(
			results_dir = test_dir,
			sim_results = results,
			comments = 'test run'
		)
# -
