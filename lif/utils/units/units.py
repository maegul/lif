from __future__ import annotations
from typing import Union, Iterable, Tuple, TypeVar, Generic
from dataclasses import dataclass, field

import numpy as np
import math
PI = math.pi

# generic values ... share arithmetic operators
val_gen = TypeVar('val_gen', float, np.ndarray)

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
class Time(Generic[val_gen]):

    value: val_gen
    unit: str = 's'
    _s: int = field(default=1, init=False)
    _ms: float = field(default=10**-3, init=False)  # magnitude of this unit in base units
    _us: float = field(default=10**-6, init=False)

    @property
    def s(self) -> val_gen:
        return (self.value * getattr(self, f'_{self.unit}')) / self._s

    @property
    def ms(self) -> val_gen:
        return (self.value * getattr(self, f'_{self.unit}')) / self._ms

    @property
    def us(self) -> val_gen:
        return (self.value * getattr(self, f'_{self.unit}')) / self._us



@dataclass(frozen=True)
class ArcLength(Generic[val_gen]):
    """Length as an angle from an origin

    value: length
    unit: deg|min|sec (degrees, minutes, seconds)
    """

    value: val_gen
    # value: Union[float, np.ndarray]
    unit: str = 'deg'
    _deg: int = field(default=1, init=False, repr=False)
    _min: float = field(default=1/60, init=False, repr=False)  # 1 min -> 1/60 degs
    _sec: float = field(default=1/(60*60), init=False, repr=False)

    @property
    def deg(self) -> val_gen:
        return (self.value * getattr(self, f"_{self.unit}")) / self._deg

    @property
    # def min(self) -> Union[float, np.ndarray]:
    def min(self) -> val_gen:
        return (self.value * getattr(self, f"_{self.unit}")) / self._min

    @property
    def sec(self) -> val_gen:
        return (self.value * getattr(self, f"_{self.unit}")) / self._sec

    @classmethod
    # def mk_multiple(cls, multi_vals: Iterable[float], unit: str) -> Tuple[ArcLength, ...]:
    def mk_multiple(
            cls, multi_vals: Iterable[val_gen],
            unit: str) -> Tuple[ArcLength[val_gen], ...]:
        "Return multiple ArcLength instances from an iterable of floats"

        return tuple(cls(value=val, unit=unit) for val in multi_vals)


# not sure this is a good idea or makes sense ... ?
@dataclass(frozen=True)
class TempFrequency(Generic[val_gen]):

    value: val_gen
    unit: str = 'hz'
    _hz: int = field(default=1, init=False)
    _w: float = field(default=1/(2*PI), init=False)

    @property
    def hz(self) -> val_gen:
        return (self.value * getattr(self, f'_{self.unit}')) / self._hz

    @property
    def w(self) -> val_gen:
        return (self.value * getattr(self, f'_{self.unit}')) / self._w


@dataclass(frozen=True)
class SpatFrequency(Generic[val_gen]):

    value: val_gen
    unit: str = 'cpd'
    _cpd: int = field(default=1, init=False)
    _cpm: float = field(default=60, init=False)  # 1 cpm -> 60 cpd
    _cpd_w: float = field(default=1/(2*PI), init=False)

    @property
    def cpd(self) -> val_gen:
        return (self.value * getattr(self, f'_{self.unit}')) / self._cpd

    @property
    def cpm(self) -> val_gen:
        return (self.value * getattr(self, f'_{self.unit}')) / self._cpm

    @property
    def cpd_w(self) -> val_gen:
        return (self.value * getattr(self, f'_{self.unit}')) / self._cpd_w
