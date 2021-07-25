"""Manage settings: declaring default, finding custom"""

from __future__ import annotations
from typing import Dict, Union, Type, TypeVar
from pathlib import Path
import json

from dataclasses import replace, dataclass

SETTINGS_FILE_NAME = Path('.lif_hws')
SIMULATION_PARAMS_FILE_NAME = Path('.lif_stim_params')


class ParamsObj:
    # just in case its helpful to know where these settings objs are from
    _settings_module_path = __file__


# generic for parameter objs
Param_T = TypeVar('Param_T', bound=ParamsObj)


@dataclass
class Settings(ParamsObj):
    "General application settings object"
    data_dir: str = "~/.lif_hws_data"


@dataclass
class SimulationParams(ParamsObj):
    "General parameters for simulations"
    spat_filt_sd_factor: float = 5


def find_update_parameters_file(
        params_obj: Param_T,
        cwd: Path, file_name: Path) -> Param_T:

    # Look for settings/config file in cwd and all parents and then home
    # if found update default settings with a dataclass replace
    # as replace raises error on replacing undeclared attribute, must conform
    for parent in (cwd, *cwd.parents, Path().home()):
        config_file = parent/file_name
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
    settings, cwd, SETTINGS_FILE_NAME)

simulation_params = find_update_parameters_file(
    simulation_params, cwd, SIMULATION_PARAMS_FILE_NAME)


def get_data_dir() -> Path:
    return Path(settings.data_dir).expanduser()
