# from typing import Dict, Union

from .core import add_conversions


@add_conversions
class Time:
    """Time quantites in seconds with properties for units


    """

    factors = {
        's': 1,
        'ms': 10**-3,
        'us': 10**-6
    }
