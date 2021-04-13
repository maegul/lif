from __future__ import annotations
from typing import Union, Iterable, Tuple
from dataclasses import dataclass, field

import numpy as np
import math
PI = math.pi

# from .core import add_conversions


# @add_conversions
# class Time:
#     """Time quantites in seconds with properties for units


#     """

#     factors = {
#         's': 1,
#         'ms': 10**-3,
#         'us': 10**-6
#     }

# > Direct Class (dataclass)
# more work per unit, but direct and easy to type
# And generates tab completion for the properties (as coded explicitly)


@dataclass(frozen=True)
class Time:

    value: Union[int, float, np.ndarray]
    unit: str = 's'
    _s: int = field(default=1, init=False)
    _ms: float = field(default=10**-3, init=False)  # magnitude of this unit in base units
    _us: float = field(default=10**-6, init=False)

    @property
    def s(self) -> float:
        return (self.value * getattr(self, f'_{self.unit}')) / self._s

    @property
    def ms(self) -> float:
        return (self.value * getattr(self, f'_{self.unit}')) / self._ms

    @property
    def us(self) -> float:
        return (self.value * getattr(self, f'_{self.unit}')) / self._us


@dataclass(frozen=True)
class ArcLength:
    """Length as an angle from an origin

    value: length
    unit: deg|min|sec (degrees, minutes, seconds)
    """

    value: Union[int, float, np.ndarray]
    unit: str = 'deg'
    _deg: int = field(default=1, init=False, repr=False)
    _min: float = field(default=1/60, init=False, repr=False)  # 1 min -> 1/60 degs
    _sec: float = field(default=1/(60*60), init=False, repr=False)

    @property
    def deg(self) -> float:
        return (self.value * getattr(self, f"_{self.unit}")) / self._deg

    @property
    def min(self) -> float:
        return (self.value * getattr(self, f"_{self.unit}")) / self._min

    @property
    def sec(self) -> float:
        return (self.value * getattr(self, f"_{self.unit}")) / self._sec

    @classmethod
    def mk_multiple(cls, multi_vals: Iterable[float], unit: str) -> Tuple[ArcLength, ...]:
        "Return multiple ArcLength instances from an iterable of floats"

        return tuple(cls(value=val, unit=unit) for val in multi_vals)


# not sure this is a good idea or makes sense ... ?
@dataclass(frozen=True)
class TempFrequency:

    value: Union[int, float, np.ndarray]
    unit: str = 'hz'
    _hz: int = field(default=1, init=False)
    _w: float = field(default=1/(2*PI), init=False)

    @property
    def hz(self) -> Union[float, np.ndarray]:
        return (self.value * getattr(self, f'_{self.unit}')) / self._hz

    @property
    def w(self) -> Union[float, np.ndarray]:
        return (self.value * getattr(self, f'_{self.unit}')) / self._w


@dataclass(frozen=True)
class SpatFrequency:

    value: Union[int, float, np.ndarray]
    unit: str = 'cpd'
    _cpd: int = field(default=1, init=False)
    _cpm: float = field(default=60, init=False)  # 1 cpm -> 60 cpd
    _cpd_w: float = field(default=1/(2*PI), init=False)

    @property
    def cpd(self) -> Union[float, np.ndarray]:
        return (self.value * getattr(self, f'_{self.unit}')) / self._cpd

    @property
    def cpm(self) -> Union[float, np.ndarray]:
        return (self.value * getattr(self, f'_{self.unit}')) / self._cpm

    @property
    def cpd_w(self) -> Union[float, np.ndarray]:
        return (self.value * getattr(self, f'_{self.unit}')) / self._cpd_w
