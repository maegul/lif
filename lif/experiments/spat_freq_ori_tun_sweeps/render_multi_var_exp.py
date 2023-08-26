# +
from typing import Sequence
from pathlib import Path
import itertools
from string import Template

# -

# # Base script template

# +
base_dir = Path('.')
base_script = base_dir / 'exp_base_large_instance_32_cores_synchrony.py'
# base_script = base_dir / 'exp_base.py'
target_dir = base_dir / 'scripts'
if not target_dir.exists():
	target_dir.mkdir()

# -

# # Variables

# templating variables to be used for substitutions:
	# ORI_BIAS_ORIENTATION: average orientation of LGN cells
	# ORI_BIAS_ORIENTATION_CV: Circular variance (ie variation) of len orientation biases
	# SPREAD_RATIO: ratio of spread
	# N_CELLS: number of LGN cells


# +
variables = {
	"ORI_BIAS_ORIENTATION_CV": [0, 0.2, 0.4, 0.6, 0.8, 1],
	"SPREAD_RATIO": [1, 2, 3, 5, 7.5, 10],
	# "ORI_BIAS_ORIENTATION": 0,
	# 0, 15, 30, 45, 60, 75, 90
	"ORI_BIAS_ORIENTATION": [75, 90],
	"N_CELLS": 30,

	# synchrony params ... COMMENT OUT
	"USE_SYNCHRONY": True,
	# 3, 5, 7.5, 10, 20
	"JITTER_TIME": 3  # ms
	}

all_individual_variables = tuple(
		[(key, v) for v in values]
			if isinstance(values, Sequence) else
		[(key, values)]
		for key, values in variables.items()
	)

all_combinations = tuple(
	dict(product)
	for product in itertools.product(*all_individual_variables)
	)
# -

# # Template substitution and write

# +
template = Template(base_script.read_text())
# -


# +
for test in all_combinations:
	new_template = template.substitute(test)
	new_path_stem = '_'.join(f'{var}_{val}' for var, val in test.items())

	# print(new_path_stem)

	new_path = (
			target_dir /
			Path(f"{base_script.stem}_{new_path_stem}{base_script.suffix}")
		)

	# print(new_path, '\n')

	new_path.write_text(new_template)
# -

