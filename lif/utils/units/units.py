"""
Objects for managing quantities as units in particular dimensions (eg, time)

## General interface for all units

* Instances represent a particular value in either the base or a specified unit
* Properties provide instance's values after having converted to the unit that the property represents
"""

from __future__ import annotations
from typing import Callable, Union, Iterable, Tuple, TypeVar, Type, Generic
from dataclasses import dataclass, field

import numpy as np
import math
PI = math.pi

scalar = Union[int, float]
val_gen = TypeVar('val_gen', scalar, np.ndarray)

# > Direct Class (dataclass)
# more work per unit, but direct and easy to type
# And generates tab completion for the properties (as coded explicitly)

unit_bc = TypeVar('unit_bc', bound='_UnitBC')

@dataclass(frozen=True)
class _UnitBC(Generic[val_gen]):

    value: val_gen
    unit: str

    # For easier programmatic access of properties: self['UNIT']
    def __getitem__(self, unit: str) -> val_gen:
        """Overload to allow unit conversion without accessing the property,
        but instead only with the name of the desired unit as a string.

        Useful for programatic manipulation, as the unit of any units object
        is accessible as a string as an attribute.

        Examples:
            >>> age, delta = Time(10, 's'), Time(5, 'us')
            >>> age['ms']
            10000.0
            >>> age.ms
            10000.0
            >>> # Get age in same units as delta
            >>> age[delta.unit]
            10000000.0
        """
        return getattr(self, unit)

    def _convert(self, new_unit: str) -> val_gen:
        """Generic conversion method, converts to new_unit from instantiated unit

        Engine of the units system in this module. **Each unit class must defined
        in such a way as to ensure that this method works properly**.

        IE - must contain private attributes (`"_NAME"`) representing the various
        available units that each have as a value the size of that unit relative
        to the base unit.
        These attributes can then be used in properties/methods that need only call
        this method (after inheriting) by passing the desired unit as the name of the
        attribute without the leading underscore (`_convert('ms')` for `'_ms'`)
        """
        return (self.value * self[f'_{self.unit}']) / self[f'_{new_unit}']

    def as_new_unit(self: unit_bc, new_unit: str) -> unit_bc:
        "Create new object of same type but in a different unit"

        unit_type = type(self)
        new = unit_type(self._convert(new_unit), new_unit)

        return new

    def in_same_units_as(self: unit_bc, other: unit_bc) -> unit_bc:
        "Re-instantiate in same base units as other"

        return self.as_new_unit(other.unit)


@dataclass(frozen=True)
class Time(_UnitBC[val_gen]):
    "Time in s, ms or us (micro)"

    value: val_gen
    unit: str = 's'
    _base_unit: str = field(default='s', init=False, repr=False)
    _s: int = field(default=1, init=False, repr=False)
    _ms: float = field(default=10**-3, init=False, repr=False)  # magnitude in base units
    _us: float = field(default=10**-6, init=False, repr=False)

    @property
    def base(self) -> val_gen:
        "Base unit: seconds (s)"
        return self._convert(self._base_unit)
        # return self.s

    @property
    def s(self) -> val_gen:
        return self._convert('s')

    @property
    def ms(self) -> val_gen:
        return self._convert('ms')

    @property
    def us(self) -> val_gen:
        return self._convert('us')


@dataclass(frozen=True)
class ArcLength(_UnitBC[val_gen]):
    """Length as an angle from an origin

    value: length
    unit: deg|mnt|sec|rad (degrees, minutes, seconds, radians), default deg
    """

    value: val_gen
    # value: Union[float, np.ndarray]
    unit: str = 'deg'
    """The unit the (initial) value attribute is in"""
    _base_unit: str = field(default='deg', init=False, repr=False)
    _deg: int = field(default=1, init=False, repr=False)
    _mnt: float = field(default=1/60, init=False, repr=False)  # 1 min -> 1/60 degs
    _sec: float = field(default=1/(60*60), init=False, repr=False)
    _rad: float = field(default=180/PI, init=False, repr=False)  # 1 radian -> 180/pi degs

    @property
    # use when in doubt (de facto conventional unit)
    def base(self) -> val_gen:
        """Base unit: degrees (deg)
        Use when in doubt (de facto conventional unit)
        """
        return self._convert(self._base_unit)
        # return self.deg

    @property
    def deg(self) -> val_gen:
        return self._convert('deg')

    @property
    # def min(self) -> Union[float, np.ndarray]:
    def mnt(self) -> val_gen:
        return self._convert('mnt')

    @property
    def sec(self) -> val_gen:
        return self._convert('sec')

    @property
    def rad(self) -> val_gen:
        "Angle or arclength in radians"
        return self._convert('rad')

    @classmethod
    # def mk_multiple(cls, multi_vals: Iterable[float], unit: str) -> Tuple[ArcLength, ...]:
    def mk_multiple(
            cls, multi_vals: Iterable[val_gen],
            unit: str) -> Tuple[ArcLength[val_gen], ...]:
        "Return multiple ArcLength instances from an iterable of floats"

        return tuple(cls(value=val, unit=unit) for val in multi_vals)


# not sure this is a good idea or makes sense ... ?
@dataclass(frozen=True)
class TempFrequency(_UnitBC[val_gen]):
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
        "Hertz: Cycles per second"
        return self._convert('hz')

    @property
    def w(self) -> val_gen:
        "Angular Frequency: radians per second (1Hz = 1cyc/s = 2pi/s)"
        return self._convert('w')


@dataclass(frozen=True)
class SpatFrequency(_UnitBC[val_gen]):
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
    _cpd_w: float = field(default=1/(2*PI), init=False, repr=False)  # rad/deg -> 1/2pi cpd

    @property
    def base(self) -> val_gen:
        "Base unit: cycles per degree (cpd)"
        return self._convert(self._base_unit)
        # return self.cpd

    @property
    def cpd(self) -> val_gen:
        return self._convert('cpd')

    @property
    def cpm(self) -> val_gen:
        return self._convert('cpm')

    @property
    def cpd_w(self) -> val_gen:
        "Spatial Angular Frequency: radians per degree (1cpd = 1cyc/deg = 2pi rad/deg"
        return self._convert('cpd_w')
