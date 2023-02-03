"""Manage settings: declaring default, finding custom

* Allows for configuration files to be located in the current working directory or in the parents thereof.
* And, for these configuration files to override defaults which are defined here.
* Settings/configuration are accessed in code by importing this module (`utils.settings`).
* Configuration values are available as attributes directly on the module itself.

```python
import utils.settings  # adjust as necessary

default_sd_factor = settings.simulation_params.spat_filt_sd_factor
```

* Default configurations are created as subclasses of [the base ParamsObj][utils.settings.ParamsObj] such as [SimulationParams][utils.settings.SimulationParams]
* These are then assigned to variables that global to the module and named by converting camel case to lowercase `snake_case`.
"""

from __future__ import annotations
from typing import Dict, Union, Type, TypeVar
from pathlib import Path
import json

from dataclasses import replace, dataclass, asdict

# from lif.utils import data_objects as do
# from lif.receptive_field.filters import contrast_correction as cont_cor


# > config file names
SETTINGS_FILE_NAME = Path('.lif_hws')
SIMULATION_PARAMS_FILE_NAME = Path('.lif_sim_params')


# > Parameters objects
class ParamsObj:
    # just in case its helpful to know where these settings objs are from
    _settings_module_path = __file__
    _file_name: Path


# generic for parameter objs
Param_T = TypeVar('Param_T', bound=ParamsObj)


# > Default settings are parameters in these classes

@dataclass
class Settings(ParamsObj):
    "General application settings object"
    _file_name: Path = SETTINGS_FILE_NAME
    data_dir: str = "~/.lif_hws_data"


@dataclass
class SimulationParams(ParamsObj):
    "General parameters for simulations"
    _file_name: Path = SIMULATION_PARAMS_FILE_NAME
    spat_filt_sd_factor: float = 5
    "By how many times the spatial coordinates should be larger than the largest Std Dev parameter"
    temp_filt_n_tau: float = 10
    "By how many times the temporal coordinates should be larger than the time constant"
    # contrast_params: do.ContrastParams = cont_cor.ON

    @property
    def contrast_params(self):
        "Default contrast params defining the default contrast curve"

        # from lif.utils import data_objects as do
        from lif.receptive_field.filters import contrast_correction as cont_cor

        return cont_cor.ON


# > updating default settings from config file

# >> update function

def find_update_parameters_file(
        params_obj: Param_T,
        cwd: Path) -> Param_T:
    """Update provided parameters object from config file

    If no config file can be found in cwd and parents, original
    object is returned.
    """

    # Look for settings/config file in cwd and all parents and then home
    # if found update default settings with a dataclass replace
    # as replace raises error on replacing undeclared attribute, must conform
    for parent in (cwd, *cwd.parents, Path().home()):
        config_file = parent/params_obj._file_name
        if not config_file.exists():
            continue
        else:
            # currently only supports flat settings
            # don't know how to do recursive here
            params_obj = replace(
                params_obj,
                **json.loads(config_file.read_text())
                )

    return params_obj


# >> Doing the actual updating and setting attributes on the module

cwd = Path.cwd()

# these attributes are intended to be available on the module directly
# allowing accessing settings to be possible immediately on import
# with `settings.simulation_params.PARAMETER`
settings = Settings()
simulation_params = SimulationParams()

# update from file
settings = find_update_parameters_file(settings, cwd)
simulation_params = find_update_parameters_file(simulation_params, cwd)


# > Utilities for writing out default config files

def get_data_dir() -> Path:
    return Path(settings.data_dir).expanduser()


def write_default_params_file(
        location: Path,
        default_class: Type[Param_T],
        overwrite: bool = False):

    params_obj = default_class()
    new_file = location / params_obj._file_name

    if new_file.exists() and not overwrite:
        raise ValueError(f'Params file of type {default_class} already exists at {location}')

    with new_file.open() as f:
        json.dump(params_obj, f)


def write_default_settings(location: Path, overwrite: bool = False):
    "Write out default values of settings to json file"

    write_default_params_file(location, Settings, overwrite)


def write_default_simulation_params(location: Path, overwrite: bool = False):
    "Write out default values of simulatin parameters to json file"

    write_default_params_file(location, SimulationParams, overwrite)
