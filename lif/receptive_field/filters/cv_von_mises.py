"""
Utility functions for calculating circular variance and von mises distribution functions
"""
from typing import Tuple, cast, Callable, Optional, List
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

from ...utils.units.units import ArcLength, SpatFrequency, val_gen
from ...utils import data_objects as do, exceptions as exc
from . import filter_functions as ff


def circ_var(r: np.ndarray, theta: ArcLength[np.ndarray]) -> float:
    """Circular Variance of given magnitudes (r) at given angles (theta)

    Args:
        r: Magnitudes of the vectors
        theta: angles of the vectors

    Returns:
        circular variance: with 0 being radially symmetrical (circular) and 1 axial

    Notes:
         Based on definition in Ringach (2000) (0 most circular, 1 most axial)
    """
    x: float = np.sum((r * np.exp(1j * 2 * theta.rad))) / np.sum(r)

    # return (1 - np.abs(x))
    return abs(x)


def von_mises(
        x: ArcLength[val_gen],
        a: float = 1, k: float = 0.5, phi: float = np.pi / 2) -> val_gen:
    """von mises distribution values for `x` for given params `a`,`k` and `phi`

    Args:
        x: angle or array of angles for which the distribution's values are generated
        a: amplitude ... value at the "*preferred*" angle
        k: "*width*" of the distribution, higher values -> *tighter* distribution
        phi: "*preferred anble*", like the mean of a guassian.

    Notes:
        Based on Swindale (1998)

    Examples:
        >>> von_mises(ArcLength(90, 'deg'))
        1.0
        >>> von_mises(ArcLength(90, 'deg'), a=13)
        13.0
        >>> cvvm.von_mises(ArcLength(np.array([10, 180+10]), 'deg'))
        array([0.61574451, 0.61574451])

    """
    curve = a * (np.exp(k * (np.cos(phi - x.rad)**2 - 1)))
    curve = cast(val_gen, curve)  # np functions don't pick up on generic, must cast

    return curve


def generate_von_mises_k_to_circ_var_vals(
        n_angles: int = 8, n_vals: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Equivalent values for the `k` parameter of a `von mises` to its circular variance

    Two arrays returned where the `nth` value in one array corresponds to the `nth` in
    the other.

    Returns:
        k values

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
                # ... np.float64 == float
                k=k_val  # type: ignore
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


# > Methods for generating orientation biased spatial filters

def mk_ori_biased_sd_factors(ratio: float) -> Tuple[float, float]:
    """Generate factors for altering the `sd` values of a 2D spatial DOG filter

    Presuming that the pre-defined (radially symmetric) `sd` should be
    and remain the  average, the generated ratio will maintain
    an average of `v` and `h` that is the same as the pre-defined `sd` value.

    Args:
        ratio: Desired value of `a/b` where `a`, `b` <-> `major`, `minor` axes
            the of spatial filter
    Returns:
        axis factors (a,b) to multiple actual `sd` values by to create the orientation bias.

    Constraints:
        * Average maintained: `a + b = 2`
        * Ratio created: `a/b = ratio` (ie, `a` is bigger)
    """

    a = (2*ratio)/(ratio+1)
    b = 2 - a

    return a, b


def mk_even_semi_circle_angles(n_angles: int = 8) -> ArcLength[np.ndarray]:

    angles = ArcLength(np.linspace(0, 180, n_angles, False), 'deg')

    return angles


def find_null_high_sf(sf_params: do.DOGSpatFiltArgs) -> SpatFrequency[float]:
    """Find spatial frequency at which DOG filter reponse is zero"""

    f = 0
    sf_min = False

    x_freq = SpatFrequency(0)
    for f in range(1000):  # 1000 cpd is too much, unrealistically high resolution!
        # 0 for x and n for y just means oriented at 0/180 degs
        y_freq = SpatFrequency(f, unit='cpd')
        r = ff.mk_dog_sf_ft(x_freq, y_freq, sf_params)
        sf_min = np.isclose(r, 0)
        if sf_min:
            break

    if (not sf_min):  # no spat freq limit was found
        raise exc.FilterError('Could not find spat-freq that elicits a zero resp for sf_params')

    return SpatFrequency(f, 'cpd')


def mk_high_density_spat_freqs(
        sf_params: do.DOGSpatFiltArgs, limit_factor: float = 5) -> SpatFrequency[np.ndarray]:
    """Uses sf parameters to estimate a good upper limit on spat_freqs

    For use with orientation bias sd ratio estimation.
    Key is to find an upper limit that is higher than the highest sf an orientation biased
    DOG is going to respond to.
    Currently, the approach is to find the spat_freq to which the sf response is zero
    and multiply this by 5.

    The returned array is from 0 to this upper limit with 1000 steps.
    """

    n = find_null_high_sf(sf_params)

    spat_freq_limit = n.base * limit_factor

    return SpatFrequency(np.linspace(0, spat_freq_limit, 1000))


def circ_var_sd_ratio_naito(
        ratio: float, sf_params: do.DOGSpatFiltArgs,
        angles: ArcLength[np.ndarray], spat_freqs: SpatFrequency[np.ndarray]
        ) -> Optional[float]:
    """Calculate circ_var of spat filt if v and h sds have provided `ratio`.

    Args:
        ratio:

    Uses definition of ori bias from Naito (2013).
    IE: Measure circ var at a high spatial frequency (higher than preferred)
    where the response is 50% of that to the preferred frequency.

    Uses mk_ori_biased_sd_factors to convert ratio to sd factors
    """

    # make new sf parameters with prescribed ratio
    new_sf_params = sf_params.mk_ori_biased_duplicate(
            *mk_ori_biased_sd_factors(ratio)
        )

    # get spatial freq resp curve for vertical grating (drifiting along 0 deg vector)
    # vertical as mk_ori_biased_sd_factors makes vertically elongated sd factors
    spat_freq_resp_v = ff.mk_dog_sf_ft(
            *ff.mk_sf_ft_polar_freqs(ArcLength(0), spat_freqs),
            new_sf_params
        )

    # find spat_freq for naito definition of how to measure ori biases
    # IE spat-freq at which the cell's response to the preferred ori is 50% of peak resp
    peak_resp = spat_freq_resp_v.max()
    peak_resp_idx = spat_freq_resp_v.argmax()
    # all idxs where response is 50% of peak or lower
    threshold_resp_idxs = (spat_freq_resp_v[peak_resp_idx:] < (0.5 * peak_resp))

    # if no such responses, can't define ori biases with naito definition at this ratio
    # with the provided spat_freqs, maybe increase spat-freq limit factor
    if threshold_resp_idxs.sum() == 0:
        return None

    # first/lowest spat_freq at or below the naito definition
    first_threshold_resp_idx = threshold_resp_idxs.nonzero()[0][0]

    threshold_spat_freq = SpatFrequency(spat_freqs.base[peak_resp_idx+first_threshold_resp_idx])

    sf_x, sf_y = ff.mk_sf_ft_polar_freqs(angles, threshold_spat_freq)
    resp = ff.mk_dog_sf_ft(sf_x, sf_y, new_sf_params)

    circ_var_value = circ_var(resp, angles)

    return circ_var_value

# >> SD Ratio Methods and module attribute
# bit hacky here ... going to create a module attribute for getting
# desired sd-ratio-method functions (and checking if it's available)

# circ var sd functions should match this type
_circ_var_sd_ratio_method_type = Callable[
        [float, do.DOGSpatFiltArgs, ArcLength[np.ndarray], SpatFrequency[np.ndarray]],
        Optional[float]
    ]

@dataclass(frozen=True)
class _CircVarSDRatioMethods:
    "Lookup of available circ_var DOG sd ratio estimation functions"
    naito: _circ_var_sd_ratio_method_type = circ_var_sd_ratio_naito

    def get_method(self, method: str) -> _circ_var_sd_ratio_method_type:
        return self.__getattribute__(method)

    @classmethod
    def _all_methods(cls) -> List[str]:
        "to check if a method is available"
        return list(cls.__dataclass_fields__.keys())

circ_var_sd_ratio_methods = _CircVarSDRatioMethods()
