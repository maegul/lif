"""Making temporal and spatial filters"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, OptimizeResult

from . import data_objects as do, filter_functions as ff
from ...utils.units.units import SpatFrequency, TempFrequency

PI: float = np.pi  # type: ignore


# from ...utils import settings

# > Temp Filters

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
        data: do.TempFiltData,
        x0: do.TQTempFiltParams = do.TQTempFiltParams.from_iter([20, 16, 4*2*PI, 0.24]),
        # x0: list = [20, 16, 4*2*PI, 0.24],
        bounds: Optional[Tuple[do.TQTempFiltParams, do.TQTempFiltParams]] = None
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
    guesses: np.ndarray = x0.array()


    # bounds
    opt_bounds_arg: Tuple[np.ndarray, np.ndarray]
    if bounds is not None:
        opt_bounds_arg = (bounds[0].array(), bounds[1].array())
    else:
        max_amplitude = max(data.amplitudes)  # type: ignore
        max_amplitude: float
        opt_bounds_arg = (
            do.TQTempFiltParams(
                amplitude=0, arguments=do.TQTempFiltArgs(tau=1, w=PI, phi=0)
                ).array(),  # mins
            # amp: double to 1 then 3 times max data
            # tau: 3 taus ~ 95% ... 100ms ... no lower level visual neuron should be longer?!
            # w: 100 (*2pi) ... 100 hz ... half-wave at 0.005 seconds (near 3* min tau)
            # phi: max phase is 2pi
            do.TQTempFiltParams(
                amplitude=max_amplitude*2*3, arguments=do.TQTempFiltArgs(
                    tau=100, w=100*2*PI, phi=2*PI)
                ).array()  # maxes
            )

    opt_res = least_squares(
        _tq_ft_wrapper, guesses, bounds=opt_bounds_arg,
        kwargs=dict(freqs=data.frequencies, amplitude_real=data.amplitudes))

    return opt_res


def make_tq_temp_filt(parameters: do.TempFiltParams) -> do.TQTempFilter:

    optimised_result = _fit_tq_temp_filt(parameters.data)

    assert optimised_result.success is True, 'optimisation is not successful'

    params = do.TQTempFiltParams.from_iter(data=optimised_result.x)

    temp_filt = do.TQTempFilter(
        source_data=parameters,
        parameters=params,
        optimisation_result=optimised_result
        )

    return temp_filt


# > Spatial Filters

def _dog_ft_wrapper(
        x: np.ndarray, freqs: SpatFrequency, amplitude_real: np.ndarray) -> np.ndarray:
    "wrap mk_dog_rf and subtract actual values from produced"

    cent_a, cent_sd, surr_a, surr_sd = x  # must match order in do.DOGSpatFiltArgs1D.array()

    # if cent_a < surr_a:  # cent should be greater than surr (some DC)
    #     return np.ones_like(amplitude_real) * 1e8
    # if cent_a > surr_a*3:  # cent shouldn't be too much greater (some band bass)
    #     return np.ones_like(amplitude_real) * 1e8
    # if cent_sd > surr_sd:
    #     return np.ones_like(amplitude_real) * 1e8

    cent_ft = ff.mk_gauss_1d_ft(freqs, amplitude=cent_a, sd=cent_sd)
    surr_ft = ff.mk_gauss_1d_ft(freqs, amplitude=surr_a, sd=surr_sd)

    dog_ft = cent_ft - surr_ft

    return dog_ft - amplitude_real


def _fit_dog_ft(
        data: do.SpatFiltData,
        x0: do.DOGSpatFiltArgs1D = do.DOGSpatFiltArgs1D.from_iter(
            [1.1, 10, 0.9, 30], arclength_unit = 'min'),
        bounds: Optional[Tuple[do.DOGSpatFiltArgs1D, do.DOGSpatFiltArgs1D]] = None
        ) -> OptimizeResult:
    "use least_squares to produced optimised parameters for mk_dog_rf"

    freqs = SpatFrequency(data.frequencies, unit='cpd')

    # guesses
    guesses: np.ndarray = x0.array()

    # bounds
    if bounds is not None:
        opt_bounds_args = (bounds[0].array(), bounds[1].array())
    else:
        max_amplitude: float
        max_amplitude = max(data.amplitudes)  # type:ignore
        max_rf_width: float = 100  # could be objectively picked ??
        opt_bounds_args = (
            do.DOGSpatFiltArgs1D(
                cent=do.Gauss1DSpatFiltParams(amplitude=0, sd=1),
                surr=do.Gauss1DSpatFiltParams(amplitude=0, sd=1)
                ).array(),
            do.DOGSpatFiltArgs1D(
                cent=do.Gauss1DSpatFiltParams(amplitude=max_amplitude*5, sd=max_rf_width),
                surr=do.Gauss1DSpatFiltParams(amplitude=max_amplitude*5, sd=3*max_rf_width)
                ).array(),
            )

    opt_res = least_squares(
        _dog_ft_wrapper, guesses, bounds=opt_bounds_args,
        kwargs=dict(freqs=freqs, amplitude_real=data.amplitudes)
        )

    return opt_res


def make_dog_spat_filt(parameters: do.SpatFiltParams) -> do.DOGSpatialFilter:

    opt_res = _fit_dog_ft(parameters.data)

    assert opt_res.success is True, 'Optmisation not successful'

    cent_a, cent_sd, surr_a, surr_sd = opt_res.x
    params = do.DOGSpatFiltArgs(
        cent=do.Gauss2DSpatFiltParams(
                amplitude=cent_a,
                arguments=do.Gauss2DSpatFiltArgs(
                    h_sd=cent_sd, v_sd=cent_sd  # symmetrical
                    )
            ),
        surr=do.Gauss2DSpatFiltParams(
                amplitude=surr_a,
                arguments=do.Gauss2DSpatFiltArgs(
                    h_sd=surr_sd, v_sd=surr_sd  # symmetrical
                    )
            )
        )

    spat_filt = do.DOGSpatialFilter(
        source_data=parameters,
        parameters=params,
        optimisation_result=opt_res
        )

    return spat_filt
