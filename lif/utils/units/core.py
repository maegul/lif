from __future__ import annotations
from typing import Union, Dict, Type, Callable, Any
from typing_extensions import Protocol

from dataclasses import dataclass, field


# > Monkey Patching approach
# works great ... creates an easy to use interface for creating multiple unit systems
# BUT ... pain to type annotate

class UnitDefinition(Protocol):
    """Class that defines unit conversions
    """
    factors: Dict[str, Union[int, float]]

    __init__: Callable

    # def __setattr__(self, name: str, value: Any) -> None:
    #     ...


class Quantity(UnitDefinition, Protocol):
    """Class that defines unit conversions
    """

    def __init__(self, quantity: Union[int, float], unit: str):
        ...

    # # So that understands callable (don't know why necessary)
    # def __call__(self, quantity: Union[int, float], unit: str):
    #     ...


def add_conversions(unit_class: Type[UnitDefinition]) -> Type[Quantity]:
    """Class decorator that adds unit facilities to simple class with factors
    
    Apply to a class that defines a single class attr "factors"
    "factors" must be a dict containing unit names in strings as keys
    and conversion factors to a base factor as values.
    The first key-value is taken to be the base factor.

    eg

    factors = {
        's': 1,
        'ms': 10**-3,
        'us': 10**-6,
    }

    Each factor defines the unit in terms of the base factor
    
    Resultant class will have an __init__(quantity, unit=default_unit)
    where default unit is the first in factors.
    For each unit, there will be a property that produces the value
    appropriate for the factor defined in factors.

    When initialised, the given quantity is converted to the 
    base/default_unit by multiplying by the provided unit/factor.
    Each property simply divides by the corresponding factor to transform
    from the base unit to that sepcified.
    """

    # base unit is first in factors
    default_unit = next(iter(unit_class.factors))

    # create unit function (could be done by inheritance)
    def init(self, quantity: Union[float, int], unit: str = default_unit):

        # always convert to base unit
        try:
            base_unit = unit_class.factors[unit]
        except KeyError as e:
            print('unit must be defined in the factors')
            print(f'unit {unit} is not in {unit_class.factors}')
            raise e
        self.quantity = quantity * base_unit
        self.default_unit = default_unit
        for unit, factor in unit_class.factors.items():
            self.__setattr__('_'+unit, factor)

    # setattr(unit_class, '__init__', init)
    unit_class.__init__ = init

    for unit in unit_class.factors:

        def prop_func(self, unit: str = unit) -> float:
            return self.quantity / self.__getattribute__('_'+unit)

        setattr(
            unit_class, 
            unit,
            property(
               prop_func 
            )
        )

    # unit_class = cast(Type[Quantity], unit_class)  # to help type checker

    return unit_class
