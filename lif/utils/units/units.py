from __future__ import annotations
from typing import Union
from dataclasses import dataclass, field

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

    value: Union[int, float]
    unit: str = 's'
    _s: int = field(default=1, init=False)
    _ms: float = field(default=10**-3, init=False)
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
