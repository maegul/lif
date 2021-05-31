from __future__ import annotations
from typing import Union, Iterable, Tuple, TypeVar, Generic
from dataclasses import dataclass, field

import numpy as np
import math
PI = math.pi

# generic values ... share arithmetic operators
val_gen = TypeVar('val_gen', int, float, np.ndarray)

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


class UnitBC(Generic[val_gen]):

    def __getitem__(self, unit: str) -> val_gen:
        return getattr(self, unit)


@dataclass(frozen=True)
class Time(UnitBC[val_gen]):

    value: val_gen
    unit: str = 's'
    _base_unit: str = field(default='s', init=False, repr=False)
    _s: int = field(default=1, init=False, repr=False)
    _ms: float = field(default=10**-3, init=False, repr=False)  # magnitude in base units
    _us: float = field(default=10**-6, init=False, repr=False)

    @property
    def base(self) -> val_gen:
        "Base unit: seconds (s)"
        return self.s

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
class ArcLength(UnitBC[val_gen]):
    """Length as an angle from an origin

    value: length
    unit: deg|mnt|sec (degrees, minutes, seconds)
    """

    value: val_gen
    # value: Union[float, np.ndarray]
    unit: str = 'deg'
    _base_unit: str = field(default='deg', init=False, repr=False)
    _deg: int = field(default=1, init=False, repr=False)
    _mnt: float = field(default=1/60, init=False, repr=False)  # 1 min -> 1/60 degs
    _sec: float = field(default=1/(60*60), init=False, repr=False)
    _rad: float = field(default=180/PI, init=False, repr=False)  # 1 radian -> 180/pi degs

    @property
    def base(self) -> val_gen:  # use when in doubt (de facto conventional unit)
        "Base unit: degrees (deg)"
        return self.deg

    @property
    def deg(self) -> val_gen:
        return (self.value * getattr(self, f"_{self.unit}")) / self._deg

    @property
    # def min(self) -> Union[float, np.ndarray]:
    def mnt(self) -> val_gen:
        return (self.value * getattr(self, f"_{self.unit}")) / self._mnt

    @property
    def sec(self) -> val_gen:
        return (self.value * getattr(self, f"_{self.unit}")) / self._sec

    @property
    def rad(self) -> val_gen:
        "Angle or arclength in radians"
        return (self.value * getattr(self, f"_{self.unit}")) / self._rad

    @classmethod
    # def mk_multiple(cls, multi_vals: Iterable[float], unit: str) -> Tuple[ArcLength, ...]:
    def mk_multiple(
            cls, multi_vals: Iterable[val_gen],
            unit: str) -> Tuple[ArcLength[val_gen], ...]:
        "Return multiple ArcLength instances from an iterable of floats"

        return tuple(cls(value=val, unit=unit) for val in multi_vals)


# not sure this is a good idea or makes sense ... ?
@dataclass(frozen=True)
class TempFrequency(UnitBC[val_gen]):
    """Temporal frequencies in units of hz (Hertz) and w (angular freq)

    w: angular frequency, represents radians per second, where 1 cycle is 2pi radians
    Thus, 1 Hz is 1 cycle per second, which is 2pi radians per second

    """

    value: val_gen
    unit: str = 'hz'
    _hz: int = field(default=1, init=False, repr=False)
    _w: float = field(default=1/(2*PI), init=False, repr=False)  # 1 rad/sec -> 1/2pi hz

    @property
    def base(self) -> val_gen:
        "Base unit: Hertz (hz) (only softly so though!)"
        return self.hz

    @property
    def hz(self) -> val_gen:
        return (self.value * getattr(self, f'_{self.unit}')) / self._hz

    @property
    def w(self) -> val_gen:
        return (self.value * getattr(self, f'_{self.unit}')) / self._w


@dataclass(frozen=True)
class SpatFrequency(UnitBC[val_gen]):
    """Spat frequencies in units of cpd, cpm (cyc per minute) and cpd_w (angular freq)

    cpd_w: angular frequency, represents radians per degree (of arclength),
    where 1 cycle is 2pi radians.
    Thus, 1 cpd is 1 cycle per degree, which is 2pi radians per second
    """

    value: val_gen
    unit: str = 'cpd'
    _base_unit: str = field(default='cpd', init=False, repr=False)
    _cpd: int = field(default=1, init=False, repr=False)
    _cpm: float = field(default=60, init=False, repr=False)  # 1 cpm -> 60 cpd
    _cpd_w: float = field(default=1/(2*PI), init=False, repr=False)

    @property
    def base(self) -> val_gen:
        "Base unit: cycles per degree (cpd)"
        return self.cpd

    @property
    def cpd(self) -> val_gen:
        return (self.value * getattr(self, f'_{self.unit}')) / self._cpd

    @property
    def cpm(self) -> val_gen:
        return (self.value * getattr(self, f'_{self.unit}')) / self._cpm

    @property
    def cpd_w(self) -> val_gen:
        return (self.value * getattr(self, f'_{self.unit}')) / self._cpd_w
