# > Imports
from typing import List, cast

import numpy as np

import plotly.express as px

from ..utils import data_objects as do
from ..utils.units.units import (
    scalar,
    ArcLength
    )
from ..receptive_field.filters import cv_von_mises as cvvm

# > Shortcuts to spatial and temporal filters

# > functions for generating cells

