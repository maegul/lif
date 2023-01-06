from dataclasses import replace

import plotly.graph_objects as go
import plotly.express as px

import numpy as np

from scipy.optimize import least_squares

from ..utils.units.units import val_gen, scalar, ArcLength

from ..utils import (
    data_objects as do,
    settings,
    exceptions as exc)


# > Contrast params

# Taken from Troyer 1998
# +
ON = do.ContrastParams(
    max_resp=53, exponent=1.2, contrast_50=0.133)
OFF = do.ContrastParams(
    max_resp=48.6, exponent=1.29, contrast_50=0.0718)
# -

# > Functions

def contrast_response(
        contrast: val_gen,
        params: do.ContrastParams
        ) -> val_gen:
    """Response for provided contrast and contrast params

    `R = (Rmax*C**(n))/(C_50**(n) + C**(n))`

    Notes:
        See:

        * Cheng, H., Chino, Y. M., Smith, E. L., Hamamoto, J., & Yoshida, K. (1995).
        Transfer characteristics of lateral geniculate nucleus X neurons in the cat: Effects of spatial frequency and contrast.
        Journal of Neurophysiology, 74(6), 2548–2557.
        * Troyer, T. W., Krukowski, A. E., Priebe, N. J., & Miller, K. D. (1998).
        Contrast-Invariant Orientation Tuning in Cat Visual Cortex: Thalamocortical Input Tuning and Correlation-Based Intracortical Connectivity.
        Journal of Neuroscience, 18(15), 5908–5927.
    """

    # ensure contrasts are fractions not percentages
    if np.any(0 > contrast > 1):
        raise ValueError('Contrast must be between 0 and 1')

    response = (
        (params.max_resp * contrast**(params.exponent))
        /
        (params.contrast_50**(params.exponent) + contrast**(params.exponent))
        )

    return response


def mk_contrast_resp_amplitude_adjustment_factor(
        base_contrast: float,
        target_contrast: float,
        params: do.ContrastParams
        ) -> float:

    base_resp = contrast_response(base_contrast, params)
    target_resp = contrast_response(target_contrast, params)

    return target_resp/base_resp
