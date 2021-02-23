"""Making temporal and spatial filters"""

from __future__ import annotations
from typing import Union, Optional, Iterable, Dict, Tuple

from dataclasses import dataclass, astuple, asdict

import numpy as np
from scipy.optimize import least_squares, OptimizeResult

from . import filter_functions as ff

numerical_iter = Union[np.ndarray, Iterable[Union[int, float]]]
PI: float = np.pi  # type: ignore


@dataclass
class RespParams:
    dc: float
    sf: float
    mean_lum: float
    contrast: float


@dataclass
class MetaData:
    author: Optional[str] = None
    year: Optional[int] = None
    title: Optional[str] = None
    doi: Optional[str] = None


# > Temp Filter

@dataclass
class TempFiltData:
    frequencies: numerical_iter
    amplitudes: numerical_iter


@dataclass
class TempFiltParams:
    data: TempFiltData
    resp_params: RespParams
    meta_data: MetaData


@dataclass
class TQTempFiltArgs:
    """fundamental args for a tq filter

    tau, w, phi
    """
    tau: float
    w: float
    phi: float

    def array(self) -> np.ndarray:
        return np.array(astuple(self))


@dataclass
class TQTempFiltParams:
    amplitude: float
    arguments: TQTempFiltArgs

    def to_array(self) -> np.ndarray:
        return np.array(
            (self.amplitude,) +
            astuple(self.arguments)
            )

    @classmethod
    def from_iter(cls, data: Iterable[float]) -> TQTempFiltParams:
        """Create object from iterable: (a, tau, w, phi)
        """
        # important ... unpacking must match order above and in
        # definition of dataclasses!
        a, tau, w, phi = data

        return cls(
            amplitude=a, 
            arguments=TQTempFiltArgs(
                tau=tau, w=w, phi=phi
                ))

    def to_flat_dict(self) -> Dict[str, float]:
        """returns flat dictionary"""
        flat_dict = asdict(self)

        flat_dict.update(flat_dict['arguments'])  # add argument params
        del flat_dict['arguments']  # remove nested dict
        return flat_dict


@dataclass
class TQTempFilter:
    source_data: TempFiltParams
    parameters: TQTempFiltParams
    optimisation_result: OptimizeResult


def _tq_ft_wrapper(
        x: np.ndarray, 
        freqs: np.ndarray, amplitude_real: np.ndarray
        ) -> np.float64:

    """Wraps mk_tq_ft() for use with least_squares()
    """

    A, tau, w, phi = x[0], x[1], x[2], x[3]  # unpack array into vars
    fit_values = A * ff.mk_tq_ft(freqs*2*PI, tau=tau, w=w, phi=phi)

    # return np.sum((fit_values - amplitude_real)**2)
    # not necessary to square and sum for least_squares?
    return fit_values - amplitude_real


def _fit_tq_temp_filt(
        data: TempFiltData,
        x0: TQTempFiltParams = TQTempFiltParams.from_iter([20, 16, 4*2*PI, 0.24]),
        # x0: list = [20, 16, 4*2*PI, 0.24],
        bounds: Optional[Tuple[TQTempFiltParams, TQTempFiltParams]] = None
        ) -> OptimizeResult:
    """Fit tq_temp_filt to given data using least_squares method

    both initial guess (x0) and bounds can be passed in.

    Default bounds are defined in code as:

    bounds = (
        np.array([0, 1, PI, 0]),  # mins
        # amp: double to 1 then 3 times max data
        # tau: 3 taus ~ 95% ... 100ms ... no lower level visual neuron should be longer?!
        # w: 100 (*2pi) ... 100 hz ... half-wave at 0.005 seconds (near 3* min tau)
        # phi: max phase is 2pi
        np.array([max_amplitude*2*3, 100, 100*2*PI, 2*PI])  # maxes
        )
    """

    # while using TQTempFiltParams as much as possible, the order of the args
    # into least_squares is most guaranteed

    # guesses
    x0 = x0.to_array()

    # bounds
    if bounds is not None:
        bounds = tuple(np.array(bound) for bound in bounds)
    else:
        max_amplitude = max(data.amplitudes)
        bounds = (
            TQTempFiltParams(
                amplitude=0, arguments=TQTempFiltArgs(tau=1, w=PI, phi=0)
                ).to_array(),  # mins
            # amp: double to 1 then 3 times max data
            # tau: 3 taus ~ 95% ... 100ms ... no lower level visual neuron should be longer?!
            # w: 100 (*2pi) ... 100 hz ... half-wave at 0.005 seconds (near 3* min tau)
            # phi: max phase is 2pi
            TQTempFiltParams(
                amplitude=max_amplitude*2*3, arguments=TQTempFiltArgs(
                    tau=100, w=100*2*PI, phi=2*PI)
                ).to_array()  # maxes
            )

    opt_res = least_squares(
        _tq_ft_wrapper, x0, bounds=bounds,
        kwargs=dict(freqs=data.frequencies, amplitude_real=data.amplitudes))

    return opt_res


def make_tq_temp_filt(parameters: TempFiltParams) -> TQTempFilter:

    optimised_result = _fit_tq_temp_filt(parameters.data)

    assert optimised_result.success is True, 'optimisation is not successful'

    params = TQTempFiltParams.from_iter(data=optimised_result.x)

    temp_filt = TQTempFilter(
        source_data=parameters,
        parameters=params,
        optimisation_result=optimised_result
        )

    return temp_filt

# plt.clf()
# plt.plot(fs, amps)
# plt.plot(fs, opt_res.x[0]*ff.mk_tq_ft(fs*2*PI, *opt_res.x[1:]))
