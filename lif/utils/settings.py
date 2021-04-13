"""Manage settings: declaring default, finding custom"""

from __future__ import annotations
from typing import Dict, Union
from pathlib import Path
import json

from dataclasses import replace, dataclass

CONFIG_FILE_NAME = Path('.lif_hws')


@dataclass
class Settings:
    data_dir: str = "~/.lif_hws_data"


settings = Settings()

cwd = Path.cwd()


# Look for settings/config file in cwd and all parents and then home
# if found update default settings with a dataclass replace
# as replace raises error on replacing undeclared attribute, must conform
for parent in (cwd, *cwd.parents, Path().home()):
    config_file = parent/CONFIG_FILE_NAME
    if not config_file.exists():
        continue
    else:
        # currently only supports flat settings
        # don't know how to do recursive here
        settings = replace(
            settings,
            **json.loads(config_file.read_text()) 
            )


def get_data_dir() -> Path:
    return Path(settings.data_dir).expanduser()
