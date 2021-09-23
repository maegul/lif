"""Manage settings: declaring default, finding custom"""

from __future__ import annotations
from typing import Dict, Union, Type, TypeVar
from pathlib import Path
import json

from dataclasses import replace, dataclass, asdict

SETTINGS_FILE_NAME = Path('.lif_hws')
SIMULATION_PARAMS_FILE_NAME = Path('.lif_stim_params')


class ParamsObj:
    # just in case its helpful to know where these settings objs are from
    _settings_module_path = __file__
    _file_name: Path


# generic for parameter objs
Param_T = TypeVar('Param_T', bound=ParamsObj)


# Default settings are parameters in these classes

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


def find_update_parameters_file(
        params_obj: Param_T,
        cwd: Path) -> Param_T:

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


cwd = Path.cwd()

settings = Settings()
simulation_params = SimulationParams()

settings = find_update_parameters_file(
    settings, cwd)

simulation_params = find_update_parameters_file(
    simulation_params, cwd)


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
