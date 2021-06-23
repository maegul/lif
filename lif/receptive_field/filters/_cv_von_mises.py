from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d

from ...utils.units.units import ArcLength, val_gen


def circ_var(r: np.ndarray, theta: ArcLength[np.ndarray]) -> float:
    """Circular Variance (Ringach (2000)) (0 most circular, 1 most axial)
    """
    x: float = np.sum((r * np.exp(1j * 2 * theta.rad))) / np.sum(r)

    # return (1 - np.abs(x))
    return abs(x)


def von_mises(
        x: ArcLength[val_gen],
        a: float = 1, k: float = 0.5,
        phi: float = np.pi / 2) -> val_gen:
    """Single von mises function (Swindale (1998))
    """
    curve: val_gen = np.exp(k * (np.cos(phi - x.rad)**2 - 1))  # type: ignore

    return a * curve


def generate_von_mises_k_to_circ_var_vals(
        n_angles: int = 8, n_vals: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """

    n_angles default is 8, as at this value the estimation of circ_var becomes stable
    16 would be better, but provides more accuracy only at finely tuned circ var values
    ( >= 0.9 ) where the estimate from 8 angles is slightly too high.
    """

    angles = ArcLength(np.linspace(0, 180, n_angles, False), 'deg')
    k_vals = np.linspace(0, 30, n_vals)

    cv_vals = np.array([
        circ_var(
            von_mises(
                angles,
                k=k_val  # type: ignore ... np.float64 == float
                ),
            angles
            )
        for k_val in k_vals
    ])

    return k_vals, cv_vals


def mk_von_mises_k_circ_var_interp_funcs(
        k_vals: np.ndarray, cv_vals: np.ndarray
        ) -> Tuple[interp1d, interp1d]:
    "Make interpolation objects for k to cv and cv to k"

    k_cv = interp1d(k_vals, cv_vals)
    cv_k = interp1d(cv_vals, k_vals)

    return k_cv, cv_k


# just generate the values, so that they're available from the module

kvals, cvvals = generate_von_mises_k_to_circ_var_vals(n_angles=8)
k_cv, cv_k = mk_von_mises_k_circ_var_interp_funcs(kvals, cvvals)

