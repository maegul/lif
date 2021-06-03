"""Making temporal and spatial filters"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, OptimizeResult

# from . import data_objects as do, filter_functions as ff
from . import filter_functions as ff
from ...utils import data_objects as do
from ...utils.units.units import ArcLength, SpatFrequency, TempFrequency, Time

PI: float = np.pi  # type: ignore
TIME_UNIT: str = 'ms'

# from ...utils import settings

# > Temp Filters


def _tq_ft_wrapper(
        x: np.ndarray, 
        freqs: TempFrequency[np.ndarray], amplitude_real: np.ndarray
        ) -> np.float64:

    """Wraps _mk_tq_ft() for use with least_squares()
    """

    # unpack array into vars
    A, tau, w, phi = x[0], x[1], x[2], x[3]
    tau = Time(tau, TIME_UNIT)
    fit_values = A * ff._mk_tqtempfilt_ft(freqs, tau=tau, w=w, phi=phi)

    # return np.sum((fit_values - amplitude_real)**2)
    # not necessary to square and sum for least_squares?
    return fit_values - amplitude_real


def _fit_tq_temp_filt(
        data: do.TempFiltData,
        x0: do.TQTempFiltParams = do.TQTempFiltParams.from_iter(
            [20, 16, 4*2*PI, 0.24], tau_time_unit=TIME_UNIT),
        bounds: Optional[Tuple[do.TQTempFiltParams, do.TQTempFiltParams]] = None
        ) -> OptimizeResult:
    """Fit tq_temp_filt to given data using least_squares method

    both initial guess (x0) and bounds can be passed in.

    NOTE: time always presumed in milliseconds

    Default bounds are defined in code as:

    opt_bounds_arg = (
        do.TQTempFiltParams(
            amplitude=0, arguments=do.TQTempFiltArgs(
                tau=Time(1, 'ms'), w=PI, phi=0)
            ).array(),  # mins
        # amp: double to 1 then 3 times max data
        # tau: 3 taus ~ 95% ... 100ms ... no lower level visual neuron should be longer?!
        # w: 100 (*2pi) ... 100 hz ... half-wave at 0.005 seconds (near 3* min tau)
        # phi: max phase is 2pi
        do.TQTempFiltParams(
            amplitude=max_amplitude*2*3, arguments=do.TQTempFiltArgs(
                tau=Time(100, 'ms'), w=100*2*PI, phi=2*PI)
            ).array()  # maxes
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
                amplitude=0, arguments=do.TQTempFiltArgs(
                    tau=Time(1, TIME_UNIT), w=PI, phi=0)
                ).array(),  # mins
            # amp: double to 1 then 3 times max data
            # tau: 3 taus ~ 95% ... 100ms ... no lower level visual neuron should be longer?!
            # w: 100 (*2pi) ... 100 hz ... half-wave at 0.005 seconds (near 3* min tau)
            # phi: max phase is 2pi
            do.TQTempFiltParams(
                amplitude=max_amplitude*2*3, arguments=do.TQTempFiltArgs(
                    tau=Time(100, TIME_UNIT), w=100*2*PI, phi=2*PI)
                ).array()  # maxes
            )

    # [><] WARNING [><] #
    # type checking doesn't penetrate this structure: _tq_ft_wrapper func args indirectly by dict
    opt_res = least_squares(
        _tq_ft_wrapper, guesses, bounds=opt_bounds_arg,
        kwargs=dict(freqs=data.frequencies, amplitude_real=data.amplitudes))

    return opt_res


def make_tq_temp_filt(parameters: do.TempFiltParams) -> do.TQTempFilter:

    optimised_result = _fit_tq_temp_filt(parameters.data)

    assert optimised_result.success is True, 'optimisation is not successful'

    params = do.TQTempFiltParams.from_iter(data=optimised_result.x, tau_time_unit=TIME_UNIT)

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

    cent_sd = ArcLength(cent_sd, 'mnt')
    surr_sd = ArcLength(surr_sd, 'mnt')
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
            [1.1, 10, 0.9, 30], arclength_unit='mnt'),
        bounds: Optional[Tuple[do.DOGSpatFiltArgs1D, do.DOGSpatFiltArgs1D]] = None
        ) -> OptimizeResult:
    "use least_squares to produced optimised parameters for mk_dog_rf"

    ### [><] WARNING [><] ###
    # units are lost in this process
    # all arclength: minuts (mnt)

    # guesses
    guesses: np.ndarray = x0.array()

    # bounds
    if bounds is not None:
        opt_bounds_args = (bounds[0].array(), bounds[1].array())
    else:
        max_amplitude: float
        max_amplitude = max(data.amplitudes)
        max_rf_width: float = 100  # could be objectively picked ??
        opt_bounds_args = (
            do.DOGSpatFiltArgs1D(
                cent=do.Gauss1DSpatFiltParams(amplitude=0, sd=ArcLength(1, 'mnt')),
                surr=do.Gauss1DSpatFiltParams(amplitude=0, sd=ArcLength(1, 'mnt'))
                ).array(),
            do.DOGSpatFiltArgs1D(
                cent=do.Gauss1DSpatFiltParams(amplitude=max_amplitude*5,
                                              sd=ArcLength(max_rf_width, 'mnt')),
                surr=do.Gauss1DSpatFiltParams(amplitude=max_amplitude*5,
                                              sd=ArcLength(3*max_rf_width, 'mnt'))
                ).array(),
            )

    opt_res = least_squares(
        _dog_ft_wrapper, guesses, bounds=opt_bounds_args,
        kwargs=dict(freqs=data.frequencies, amplitude_real=data.amplitudes)
        )

    return opt_res


def make_dog_spat_filt(parameters: do.SpatFiltParams) -> do.DOGSpatialFilter:
    """Make a DOG filter from data"""

    opt_res = _fit_dog_ft(parameters.data)

    assert opt_res.success is True, 'Optmisation not successful'

    cent_a, cent_sd, surr_a, surr_sd = opt_res.x
    cent_sd = ArcLength(cent_sd, 'mnt')
    surr_sd = ArcLength(surr_sd, 'mnt')

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
