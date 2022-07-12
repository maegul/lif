"""Functions and classes for the generation of spatial and temporal filters
"""

from dataclasses import dataclass
from typing import Optional, Union, Tuple, overload, Callable, cast, NoReturn
from textwrap import dedent

from brian2.core.variables import Variable
import numpy as np

# import matplotlib.pyplot as plt

# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui

# import scipy.stats as st

from . import cv_von_mises as cvvm
from ...utils import data_objects as do, settings, exceptions as exc
from ...utils.data_objects import ConvRespAdjParams, SpaceTimeParams
from ...utils.units.units import (
    SpatFrequency, TempFrequency, ArcLength, Time,
    val_gen, scalar)
from ...convolution import estimate_real_amp_from_f1 as est_amp


# > Globals and Settings

PI: float = np.pi
SPAT_FILT_SD_LIMIT = settings.simulation_params.spat_filt_sd_factor
TEMP_EXT_N_TAU = settings.simulation_params.temp_filt_n_tau

# > Coordinatess

# >> Spat Coords

def round_coord_to_res(
        coord: ArcLength[float], res: ArcLength[int],
        high: bool = False, low: bool = False) -> ArcLength[int]:
    """Rounds, or *"snaps"* a Spatial coord to a whole number
    multiple of the resolution

    coords returned in units of resolution and with value as an int
    (provided res is int)

    Args:
        coord: Spatial length to be rounded
        res: "*resolution*" that the coord is to be rounded to or "*snapped*" to.
            The returned coordinate will be an integer multiple of this value.
            Its value must be an integer, if not, TypeError is returned.
            This is to help ensure the returned coord is also an int.
            To aid in ensuring this,
            [SpaceTimeParams][utils.data_objects.SpaceTimeParams] has a
            post_init method to ensure that the
            [spatial res parameter][utils.data_objects.SpaceTimeParams.spat_res]
            is an integer.
        high: whether to round/snap coord to higher than initial value
        low: like `high` but lower than initial value

    Note:
        If neither `high` nor `low` is True, then closest to initial is returned

    Returns:
        Initial coord value, snapped to `res`,
        as an integer value, in same unit as res

    Examples:
        >>> round_coord_to_res(
        ...     coord=ArcLength(1.5, 'deg'), res=ArcLength(20, 'mnt'))
        ArcLength(value=80, unit='mnt')
    """

    if high and low:
        raise ValueError('high and low cannot both be True')
    if not isinstance(res.value, int):
        raise TypeError(f'res value must be int, not {type(res.value)}')

    res_val = res.value
    res_unit = res.unit
    coord_val = coord[res_unit]

    # type hint as int as a guard ... helps guarantee that the output is actually int
    # as it will be from one of these values
    # useful as can't type a unit (ie ArcLength) as int, only "scalar"
    low_val: int = int(coord_val // res_val) * res_val
    high_val: int = low_val + res_val
    rounded_val: int

    if high:
        rounded_val = high_val
    elif low:
        rounded_val = low_val
    else:
        low_diff = abs(low_val - coord_val)
        high_diff = abs(high_val - coord_val)

        rounded_val: int = (
            high_val
                if low_diff > high_diff else
            low_val
            )

    new_coord: ArcLength[int] = ArcLength(rounded_val, res_unit)

    return new_coord

    # return ArcLength(rounded_val, res_unit)


# >! Redundant?
def mk_spat_radius(spat_ext: Union[ArcLength[int], ArcLength[float]]) -> ArcLength[int]:
    """Calculate a radius that is appropriate for generating spatial coords

    Divides extent by 2 and raises to ceiling.
    Use ceil (instead of floor) to ensure that sd_limit is
    not arbitrarily cut down too far.

    Value is converted to int (through ceil) and return is always ArcLength[int]

    All done in same units as spat_ext and returns arclength in same units
    """
    # val = int(np.ceil(spat_ext.value / 2))
    # why not just round to next highest res multiple??

    ## Redundant now with rounded_spat_radius function and round_coord_to_res function??

    val = round(spat_ext.value / 2)
    spat_radius = ArcLength(val, spat_ext.unit)

    return spat_radius


def mk_rounded_spat_radius(
        spat_res: ArcLength[int], spat_ext: Union[ArcLength[int], ArcLength[float]]
        ) -> ArcLength[int]:
    """Round spat_ext to whole multiple of spat_res.
    Return value is `int` and the unit is same as `spat_res`.

    Uses `round_coord_to_res`, after dividing by 2, and rounding to the higher value

    Examples:
        >>> (0.7893 * 60) / 2  # the "radius" defined by 0.7893 degs
        23.679
        >>> mk_rounded_spat_radius(ArcLength(3, 'mnt'), ArcLength(0.7839, 'deg'))
        ArcLength(value=24, unit='mnt')
    """

    spat_radius = round_coord_to_res(
        ArcLength(spat_ext[spat_res.unit] / 2, spat_res.unit),
        spat_res, high=True
    )

    return spat_radius


def check_spat_ext_res(ext: ArcLength[int], res: ArcLength[int]):
    """Check that spatial ext and res are capable of producing appropriate coords

    That is ... Extent is a whole number multiple of resolution ...
    AND that the two are in the same units

    THE UNITS OF THE TWO should be the same to prevent errors in the generation of the coords

    Raises:
        CoodsValueError: When the units aren't the same, or extent not a whole multiple of res

    """

    if (ext.unit != res.unit):
        raise exc.CoordsValueError(
            f'Extent unit {ext.unit} and res unit {res.unit} should be same')

    check = ((ext.value % res.value) == 0)
    if not check:
        raise exc.CoordsValueError(
            f'extent {ext} not whole multiple of res {res}: ext % res = {ext.value % res.value}')


def mk_spat_coords_1d(
        spat_res: ArcLength[int] = ArcLength(1, 'mnt'),  # should be ArcLength[int] ??
        spat_ext: Union[ArcLength[int], ArcLength[float]] = ArcLength(300, 'mnt')
        ) -> ArcLength[np.ndarray]:
    """1D spatial coords symmetrical about 0 with 0 being the central discrete point
        with resolution `spat_res` and extent/width/diameter along both `x` and `y` axes
        defined by `spat_ext`.  Returned `ArcLength` is in same units as `spat_res`.

    Args:
        spat_res: Resolution of the coordinates.
        spat_ext: Extent of the coordinates, **from a edge to edge, like a diameter**.

    Returns:
        coords: newly generated coordinates with a central value of 0, **in same unit
            as `spat_res`**

    Examples:
        >>> mk_spat_coords_1d(ArcLength(2, 'mnt'), ArcLength(0.1, 'deg'))
        ArcLength(value=array([-4, -2,  0,  2,  4]), unit='mnt')

    Notes:
        Center point (value: 0) will be at index spat_ext // 2 if spat_res is 1
        Or, otherwise ... coords.size // 2,
        Or, as always square for 2d (?) ... coords.shape[0] // 2

        **Guranteed** by ensuring the radius/extent used is a whole number multiple
        of the res, and, adding to extent 1 additional unit of spatial resolution.

        This ensures that the total number of coordinates is an odd number (`2*n + 1` => odd).
        Which ensures that there is a **central** value (like 3 in 1 2 3 4 5).
        And, as the array of coords has symmetrical positive and negative extents
        (using `np.arange` `start` and `stop` paramters, eg, -10 and 10), this ensures
        that the central value is zero.

        All calculations done in units of spat_res
    """

    # to have a coord value that is zero, it is a requirement that
    # the spat_radius is a whole multiple of the spat_res
    spat_radius = mk_rounded_spat_radius(spat_res, spat_ext)
    check_spat_ext_res(ext=spat_radius, res=spat_res)

    # get radius using same units as spat_res
    coords = ArcLength(
            np.arange(
                -spat_radius.value,
                spat_radius.value + spat_res.value,
                spat_res.value
                ),
            spat_res.unit
        )

    return coords


def mk_spat_ext_from_sd_limit(
        sd: ArcLength[float],
        sd_limit: float = SPAT_FILT_SD_LIMIT
        ) -> ArcLength[int]:
    """Calculate spatial extent to adequately present full DOG spatial filter with Std Dev sd

    As sd is a radius distance, and spat_ext is diametrical, calculation is
    2 * sd_limit * sd.
    Raising to the ceiling integer of the result of (sd_limit and sd) is done before doubling
    up to an extent

    Returns Arclenth in same units as sd.
    """

    max_sd = int(np.ceil(sd_limit * sd.value))
    spat_ext = ArcLength(2 * max_sd, sd.unit)

    return spat_ext


# >> Standard Dev limited Spat Coords
# This function handles limiting spat coords by a Std Dev argument
# Main behaviour is require only one of a spat_ext or std dev arg
# and return the appropriate output or raise an error ... thus
# the overloads
@overload  # both none: Exception
def mk_sd_limited_spat_coords(
        spat_res: ArcLength[int],
        spat_ext: None,
        sd: None,
        sd_limit: float = SPAT_FILT_SD_LIMIT) -> NoReturn: ...
@overload  # both provided: Exception
def mk_sd_limited_spat_coords(
        spat_res: ArcLength[int],
        spat_ext: ArcLength[scalar],
        sd: ArcLength[scalar],
        sd_limit: float = SPAT_FILT_SD_LIMIT) -> NoReturn: ...
@overload  # only one: good
def mk_sd_limited_spat_coords(
        spat_res: ArcLength[int],
        spat_ext: None,
        sd: ArcLength[scalar],
        sd_limit: float = SPAT_FILT_SD_LIMIT) -> ArcLength[np.ndarray]: ...
@overload  # only one: good
def mk_sd_limited_spat_coords(
        spat_res: ArcLength[int],
        spat_ext: ArcLength[scalar],
        sd: None,
        sd_limit: float = SPAT_FILT_SD_LIMIT) -> ArcLength[np.ndarray]: ...

def mk_sd_limited_spat_coords(
        spat_res: ArcLength[int] = ArcLength(1, 'mnt'),
        spat_ext: Optional[ArcLength[scalar]] = None,
        sd: Optional[ArcLength[scalar]] = None,
        sd_limit: float = SPAT_FILT_SD_LIMIT) -> ArcLength[np.ndarray]:
    """Wraps `mk_spat_coords_1d` to limit extent by number of SD if provided

    Must provide either spat_ext or sd, **but not both or neither**.

    If using `sd`, it is rounded up to the ceiling of its product with `sd_limit`
    (see `mk_spat_ext_from_sd_limit`)

    Returns:
        spat_coords: 1D coordinates in same unit as spat_res (from mk_spat_coords_1d)

    Raises:
        CoordsValueError: When both spat_ext and sd are proivded or both are None
            (*must provide at least and only one*)
    """

    if (spat_ext is None) and (sd is None):
        raise exc.CoordsValueError(
            f'At least one of spat_ext or sd must be provided.  Both are: {spat_ext, sd}')
    if spat_ext and sd:
        raise exc.CoordsValueError(
            f'Must provide only one of spat_ext or sd, not both.  Both are: {spat_ext, sd}')

    # use sd if provided
    if spat_ext:
        final_spat_ext = spat_ext
    elif sd:
        final_spat_ext = mk_spat_ext_from_sd_limit(sd, sd_limit)
    # remove "possibly unbound" typing issue
    final_spat_ext: ArcLength[scalar] = cast(ArcLength[scalar], final_spat_ext)

    coords = mk_spat_coords_1d(spat_res=spat_res, spat_ext=final_spat_ext)

    return coords


def mk_spat_coords(
        spat_res: ArcLength[int] = ArcLength(1, 'mnt'),
        spat_ext: Optional[ArcLength[scalar]] = None,
        sd: Optional[ArcLength[float]] = None,
        sd_limit: float = SPAT_FILT_SD_LIMIT
        ) -> Tuple[ArcLength[np.ndarray], ArcLength[np.ndarray]]:

    '''
    Produces spatial (ie meshgrid) for
    generating RFs and stimuli
    Base units are minutes of arc and milliSeconds

    Args:
        spat_res:
            resolution, in minutes of arc, of meshgrid
        spat_ext:
            * Width and height, in minutes of arc, of spatial dimensions of meshgrid.
            * `spat_ext` is the total width (horiztontal or vertical) of the canvas
            image generated for receptive fields and stimuli.
            * The radial extent (horizontal or vertical, from center) will be
            floor(spat_ext/2), **and always centred on zero**.
            * So *the actual extent* will be `1*spat_res greater` than `spat_ext` for
            even and as specified for odd extents
            * **Cannot be provided with `sd`.  Must provide only one of `sd` or `spat_ext`.**
        sd:
            * If provided, coords will be limited to the ceiling of this multiplied by `sd_limit`.
            * See [mk_sd_limited_spat_coords][
                receptive_field.filters.filter_functions.mk_sd_limited_spat_coords]
            * **Can only be provided if `spat_ext` not provided**.
        sd_limit:
            Used if `sd` provided.  *The number of Std Deviations the coords are to be limited to*.

    Returns:
        X and Y coords as meshgrids, in same unit as `spat_res`.

    All meshgrids filled with appropriate coordinate values
    '''

    # spat_radius = np.floor(spat_ext / 2)
    # x_coords = np.arange(-spat_radius, spat_radius + spat_res, spat_res)

    res_unit = spat_res.unit  # use this as base unit for all other values

    # mk_sd_limited_spat_coords responds to arguments appropriately
    x_coords = mk_sd_limited_spat_coords(
        spat_ext=spat_ext, spat_res=spat_res,
        sd=sd, sd_limit=sd_limit)

    # treat as image (with origin at top left or upper)
    # y_cords positive at top and negative at bottom
    y_coords = ArcLength(x_coords.value[::-1], x_coords.unit)  # use same unit as x_coords

    xc: ArcLength[np.ndarray]
    yc: ArcLength[np.ndarray]

    # use res_unit
    xc, yc = (
        ArcLength(c, res_unit)
        for c
        in np.meshgrid(x_coords[res_unit], y_coords[res_unit])  # type: ignore
        )

    return xc, yc


# >> Temp Coords

@overload  # ext provided
def mk_temp_coords(
        temp_res: Time[scalar],
        temp_ext: Time[scalar],
        tau = None,
        temp_ext_n_tau: Optional[float] = None) -> Time[np.ndarray]: ...
@overload  # tau provided
def mk_temp_coords(
        temp_res: Time[scalar],
        temp_ext: None,
        tau: Time[scalar],
        temp_ext_n_tau: Optional[float] = None) -> Time[np.ndarray]: ...
@overload  # both none ... exception raised
def mk_temp_coords(
        temp_res: Time[scalar],
        temp_ext: None,
        tau: None,
        temp_ext_n_tau: Optional[float] = None) -> NoReturn: ...
@overload  # both provided ... exception raised
def mk_temp_coords(
        temp_res: Time[scalar],
        temp_ext: Time[scalar],
        tau: Time[scalar],
        temp_ext_n_tau: Optional[float] = None) -> NoReturn: ...
def mk_temp_coords(
        temp_res: Time[scalar],
        temp_ext: Optional[Time[scalar]] = None,
        tau: Optional[Time[scalar]] = None,
        temp_ext_n_tau: Optional[float] = None) -> Time[np.ndarray]:
    """Array of times with resolution temp_res and extent temp_ext or tau * temp_text_n_tau

    Returned coords are in the same unit as `temp_res`.

    Args:
        temp_res: Resolution
        temp_ext: Extent of time span (from zero to this value).
            Last temporal coordinate will in the window of this value and and this+`temp_res`
            Required if not providing a `tau`.
        tau: Time constant of a temporal filter.
            Must be provided if not provided `temp_ext`.
            The `tau` (or time constant) of temporal filter these coords will be used on.
            Can therefore define the extent of the coords by the time constant so as to
            ensure the full envelope of the filter is captured on the "canvas" of these
            coords.
            If provided, the extent of the coords is calculated from this `tau`.
        temp_ext_n_tau: Optional (along with `tau`) for controlling the extent of the coords.
            If both provided, `temp_ext = tau * temp_ext_n_tau`.
            Cannot be less than TEMP_EXT_N_TAU, as a general heuristic for ensuring a minimum capture
            of temporal filter.

    Returns:
        temp_coords: array of regular Time values from 0 to at least the defined extent
    """

    res_unit = temp_res.unit

    # exception for both ext and tau or neither
    #   V--> neither                         -|-  V--> both
    if ((temp_ext is None) and (tau is None)) or (temp_ext and tau):
        raise exc.CoordsValueError(dedent(f'''
            Only one of temp_ext or tau must be provided, not both or neither.
            Instead they are: {temp_ext, tau}'''))

    # set temp_ext according to tau and temp_ext_n_tau
    if tau:
        if (temp_ext_n_tau is None):
            temp_ext_n_tau = TEMP_EXT_N_TAU
        elif temp_ext_n_tau < TEMP_EXT_N_TAU:  # hardcode prohibition against n_tau less than TEMP_EXT_N_TAU!
            raise exc.CoordsValueError(dedent(f'''
                temp_ext_n_tau ({temp_ext_n_tau}) should be {TEMP_EXT_N_TAU} or more
                Any lower and the sum of the filter will be diminished
                by approx 0.1% or more for tq filt'''))

        temp_ext = Time(tau[res_unit] * temp_ext_n_tau, res_unit)

    # by here, either temp_ext provided or created in if-statement above
    # type checker might not get that though :)
    temp_ext = cast(Time[float], temp_ext)
    # stop set to ext + res so that max value in coords is AT LEAST the provided extent.
    t = Time(
        np.arange(
            0,
            temp_ext[res_unit] + temp_res[res_unit], # add res to stop, so max never less than ext
            temp_res[res_unit]),
        res_unit
        )

    return t


# >> Spat-Temp coords

def mk_spat_temp_coords(
        spat_res: ArcLength[int],
        temp_res: Time[scalar],
        spat_ext: Optional[ArcLength[scalar]] = None,
        temp_ext: Optional[Time[scalar]] = None,
        sd: Optional[ArcLength[float]] = None,
        sd_limit: float = SPAT_FILT_SD_LIMIT,
        tau: Optional[Time[scalar]] = None,
        temp_ext_n_tau: float = TEMP_EXT_N_TAU
        ) -> Tuple[ArcLength[np.ndarray], ArcLength[np.ndarray], Time[np.ndarray]]:
    '''
    Produces **grids** or **matrices** of spatial and temporal coordinates
    (using `np.meshgrid`) for generating RFs and stimuli.
    Base units are minutes of arc and milliSeconds

    Args:
        spat_res: resolution, in minutes of arc, of meshgrid
        temp_res: resolution, in milliseconds, of temporal dimension of meshgrid
        spat_ext: Width and height, in minutes of arc, of spatial dimensions of meshgrid
            spat_ext is the total width (horiztontal or vertical) of the stimulus
            image generated.
            radial extent (horizontal or vertical, from center) will be
            floor(spat_ext/2), and always centred on zero.
            So actual extent will be 1*spat_res greater for even and as specified
            for odd extents
        temp_ext: duration, in milliseconds, of temporal dimension
        sd: std dev of putative spatial filter
        sd_limit: if sd present, passed to mk_sd_limited_spat_coords to limit extent automatically.
            spat_ext will be ignored.  Default value is from configuration
        tau: time contant of a temporal filter, passed to mk_temp_coords to limit the extent
            according to a putative temporal filter
        temp_ext_n_tau: number of time constants to use as the extent, default from config.

    Returns:
        xc: 3D output of `np.meshgrid` `x` coord values as an `ArcLength` object,
            where dims are: `[y,x,t]` **as rows are first dim, and represent visual y-axis**.
        yc: 3D output as `xc` but with `y` coord values
        tc: 3D output as `xc` but with `t` (temporal) values

    Notes:
        `np.meshgrid` indexing is by default `"xy"`, meaning that the "first" dimension will
        represent the "second" coordinate vector that is passed to the function: the `y` in
        this case.

        Thus, to get the value at the "second" `x`, "third" `t` and "first" `y` coordinate
        location would require indexing, where `0="first"`: `[0, 1, 2]` (`y`, `x`, `t`).

        More generally, about `np.meshgrid` and the variables returned by this function,
        their purpose is to provide all the 1D coordinate vectors given to the `np.meshgrid`
        function at all the locations in the `N-dimenstional` space (where `N` is the number
        of coordinate vectors provided) implied by the 1D vectors.

        For a 2D space, one could have a 2D array with each value being a `tuple` of 2 values,
        each representing one of the dimensions.

        ```python
        [[(0,0), (0,1)],
         [(1,0), (1,1)]]
        ```

        `np.meshgrid`, and this function, return instead one array for each dimension.
        Each array has `N-dimensions` (like that above, which is 2D), and *repeats values
        along all dimensions except that to which the array corresponds*.
        This way the coordinates of each dimensions can easily be treated separately,
        as they are separate arrays or variables.
        But the indexing of each of the arrays or coordinates will be the same, such that
        for any array/coordinate `C`, `C[11, 45, 12]` will retrieve the value of the coordinate
        of the relevant dimension at the defined location (`11, 45, 12`) in the same way for
        each array/coordinate.

        Thus, the above would be ...

        ```python
        # Y coords
        [[0, 0],
         [1, 1]]

        # X coords
        [[0, 1],
         [0, 1]]
        ```

        Hopefully it is clear how the arrangement of the values both follows the X and Y-axis
        arrangement we're accustomed to (x being horizontal and y vertical)
        as well as repeats values along the irrelevant dimensions (eg, the horizontal rows for Y).
    '''

    # spat_radius = np.floor(spat_ext / 2)
    # x_coords = np.arange(-spat_radius, spat_radius + spat_res, spat_res)

    # mk_sd_limited_spat_coords responds to arguments appropriately
    x_coords = mk_sd_limited_spat_coords(
        spat_ext=spat_ext, spat_res=spat_res,
        sd=sd, sd_limit=sd_limit)

    # treat as image (with origin at top left or upper)
    # y_cords positive at top and negative at bottom
    # generated in same unit as x_coords
    y_coords = ArcLength(x_coords.value[::-1], x_coords.unit)

    t_coords = mk_temp_coords(
        temp_res=temp_res, temp_ext=temp_ext, tau=tau, temp_ext_n_tau=temp_ext_n_tau)

    _xc: np.ndarray
    _yc: np.ndarray
    _tc: np.ndarray

    # use .value and .unit of original 1d arrays to maintain the same units
    # as the 1d arrays the meshgrid is made from.
    # this way, the task of rendering coords in the appropriate units (same as resolution, say)
    # is on the core 1d functions only.
    _xc, _yc, _tc = np.meshgrid(x_coords.value, y_coords.value, t_coords.value)
    xc = ArcLength(_xc, x_coords.unit)
    yc = ArcLength(_yc, y_coords.unit)
    tc = Time(_tc, t_coords.unit)

    return xc, yc, tc


def mk_blank_coords(
        spat_res: ArcLength[int] = ArcLength(1, 'mnt'), temp_res: Time = Time(1, 'ms'),
        spat_ext: Union[ArcLength[int], ArcLength[float]] = ArcLength(300, 'mnt'),
        temp_ext: Time = Time(1000, 'ms')
        ) -> np.ndarray:
    '''
    Produces blank coords for each spatial and temporal value
    that would be ordinarily created
    Base units are minutes of arc and milliSeconds

    Parameters
    ----
    spat_res
        resolution, in minutes of arc, of meshgrid
    temp_res : float
        resolution, in milliseconds, of temporal dimension of meshgrid
    spat_ext : int
        Width and height, in minutes of arc, of spatial dimensions of meshgrid
        spat_ext is the total width (horiztontal or vertical) of the stimulus
        image generated.
        radial extent (horizontal or vertical, from center) will be
        floor(spat_ext/2), and always centred on zero.
        So actual extent will be 1*spat_res greater for even and as specified
        for odd extents
    temp_ext : int
        duration, in milliseconds, of temporal dimension

    Returns
    ----
    single meshgrid (3D: x,y,t)
    '''

    # spat_radius = np.floor(spat_ext / 2)
    # x_coords = np.arange(-spat_radius, spat_radius + spat_res, spat_res)

    x_coords = mk_spat_coords_1d(spat_res, spat_ext)
    # treat as image (with origin at top left or upper)
    # y_cords positive at top and negative at bottom
    y_coords = ArcLength(x_coords.mnt[::-1], 'mnt')

    t_coords = Time(np.arange(0, temp_ext.ms, temp_res.ms), 'ms')

    # ie, one array with appropriate size, each coordinate represented by a single value
    space: np.ndarray
    space = np.zeros((y_coords.mnt.size, x_coords.mnt.size, t_coords.ms.size))  # type: ignore

    return space


# > Spatial

def mk_gauss_1d(
        coords: ArcLength[np.ndarray],
        mag: float = 1, sd: ArcLength[float] = ArcLength(10, 'mnt')) -> np.ndarray:
    """Simple 1d gauss


    Note
    ----
    Magnitude is dependent on the units used in the calculation

    Returned "function" as units ArcLength^{-1}, which is why the amplitude
    is unit dependent
    """

    gauss_coeff = mag / (sd.mnt * (2*PI)**0.5)  # ensure integral of 1
    # gauss_1d: np.ndarray  # as coords SHOULD always be an np.array
    gauss_1d = gauss_coeff * np.exp(-coords.mnt**2 / (2 * sd.mnt**2))  # type: ignore

    return gauss_1d


def _mk_ft_freq_symmetry_factor(freqs: val_gen) -> val_gen:
    "Factor for collapsing symmetry of freqs depending on whether 0 or not"

    symm_fact = 1

    if isinstance(freqs, (float, int)):
        if freqs != 0:
            symm_fact = 2

    elif isinstance(freqs, np.ndarray):
        # one is base, add ones where freq is not zero to double
        symm_fact = np.ones_like(freqs) + (freqs != 0).astype(int)

    return symm_fact  # type: ignore


def _mk_ft_freq_symmetry_factor_2d(x_freqs: val_gen, y_freqs: val_gen) -> val_gen:
    "As with freq_symmetry_factor, but for 2d freqs, where non-zero in either requires collapse"

    x_fact = _mk_ft_freq_symmetry_factor(x_freqs)
    y_fact = _mk_ft_freq_symmetry_factor(y_freqs)

    # should be 1 only where 1 in both x and y, otherwise 2 even if x or y has a 1
    combined_fact: val_gen = np.maximum(x_fact, y_fact)  # type: ignore

    return combined_fact


def mk_gauss_1d_ft(
        freqs: SpatFrequency[val_gen],
        amplitude: float = 1,
        sd: ArcLength[float] = ArcLength(10, 'mnt'),
        collapse_symmetry: bool = False) -> val_gen:
    """Returns normalised ft, treats freqs as  cycs per minute

    Works with mk_gauss_1d

    Presumes normalised gaussian (relies on such for mk_gauss_1d).
    Thus coefficient is 1

    collapse_symmetry: whether to multiply non-zero amplitudes by 2 to remove
    the positive/negative freq symmetry of fourier transforms
    True for sinusoidal amplitudes, False for convolutional amplitudes

    freqs: cpm

    Question of what amplitude is ... arbitrary amplitude of spatial filter??
    """
    ft: val_gen
    # Units for freqs and sd don't matter, so long as they match
    # as their magnitudes change in opposite directions
    ft = amplitude * np.exp(-PI**2 * freqs.cpm**2 * 2 * sd.mnt**2)  # type: ignore

    if collapse_symmetry:
        freq_sym_factor = _mk_ft_freq_symmetry_factor(freqs.value)
        ft = ft * freq_sym_factor

    # Note: FT = T * FFT ~ amplitude of convolution
    return ft


def mk_gauss_2d(
        x_coords: ArcLength, y_coords: ArcLength,
        gauss_params: do.Gauss2DSpatFiltParams) -> np.ndarray:

    gauss_x = mk_gauss_1d(x_coords, sd=gauss_params.arguments.h_sd)
    gauss_y = mk_gauss_1d(y_coords, sd=gauss_params.arguments.v_sd)

    gauss_2d = gauss_params.amplitude * gauss_x * gauss_y

    return gauss_2d


def mk_gauss_2d_ft(
        freqs_x: SpatFrequency[val_gen], freqs_y: SpatFrequency[val_gen],
        gauss_params: do.Gauss2DSpatFiltParams,
        collapse_symmetry: bool = False) -> val_gen:
        # collapse_symmetry: bool = False) -> Union[float, np.ndarray]:
    """Analytical fourier transform of 2d gaussian defined by gauss_params


    collapse_symmetry: whether to multiply non-zero amplitudes by 2 to remove
    the positive/negative freq symmetry of fourier transforms.
    True for sinusoidal amplitudes, False for convolutional amplitudes

    Parameters
    ----


    Returns
    ----

    """

    # amplitude is 1 for 1d so that amplitude for 2d applies to whole 2d
    gauss_ft_x = mk_gauss_1d_ft(freqs_x, sd=gauss_params.arguments.h_sd, amplitude=1)
    gauss_ft_y = mk_gauss_1d_ft(freqs_y, sd=gauss_params.arguments.v_sd, amplitude=1)

    gauss_2d_ft = gauss_params.amplitude * gauss_ft_x * gauss_ft_y

    if collapse_symmetry:
        freq_sym_fact = _mk_ft_freq_symmetry_factor_2d(freqs_x.value, freqs_y.value)
        gauss_2d_ft = gauss_2d_ft * freq_sym_fact

    # Note: FT = T * FFT ~ amplitude of convolution

    # Could use cast to pretend int still possible
    # gauss_2d_ft = cast(val_gen, gauss_2d_ft)
    return gauss_2d_ft


# sf = spatial filter
def mk_dog_sf(
        x_coords: ArcLength[np.ndarray], y_coords: ArcLength[np.ndarray],
        dog_args: do.DOGSpatFiltArgs) -> np.ndarray:

    cent_gauss_2d = mk_gauss_2d(x_coords, y_coords, gauss_params=dog_args.cent)
    surr_gauss_2d = mk_gauss_2d(x_coords, y_coords, gauss_params=dog_args.surr)

    return cent_gauss_2d - surr_gauss_2d


def mk_dog_sf_ft(
        freqs_x: SpatFrequency[val_gen], freqs_y: SpatFrequency[val_gen],
        dog_args: do.DOGSpatFiltArgs, collapse_symmetry: bool = False) -> val_gen:
    """Amplitude of analytical fourier transform of dog_sf of dog_args

    Returns the amplitude of the fourier transform of the DOG Spat Filt
    defined by the parameters in dog_args.
    This fourier transform is the analytical and needs to be scaled by the
    spatial resolution employed in any convolutional context to provide
    an amplitude representative of the output of that convolution.

    IE: FT (analytical) = Res * FFT ~ amplitde of convolution

    Parameters
    ----

    collapse_symmetry: (default True)
        whether to multiply non-zero amplitudes by 2 to remove
        the positive/negative freq symmetry of fourier transforms.
        True for sinusoidal amplitudes, False for convolutional amplitudes

        Default is False, as the purpose of this func is direct calculation
        of convolutional amplitudes (or results of other extraneous processes)


    Returns
    ----

    """

    # somewhat redundant to collapse symmetry twice, but cleaner code
    cent_ft = mk_gauss_2d_ft(
        freqs_x, freqs_y, gauss_params=dog_args.cent, collapse_symmetry=collapse_symmetry)
    surr_ft = mk_gauss_2d_ft(
        freqs_x, freqs_y, gauss_params=dog_args.surr, collapse_symmetry=collapse_symmetry)

    dog_rf_ft = cent_ft - surr_ft

    # Note: FT = T * FFT ~ amplitude of convolution
    return dog_rf_ft


def mk_dog_sf_ft_1d(
        freqs: SpatFrequency[val_gen],
        dog_args: Union[do.DOGSpatFiltArgs, do.DOGSpatFiltArgs1D],
        collapse_symmetry: bool = False) -> val_gen:
    """Make 1D Fourier Transform of DoG RF but presume radial symmetry and return only 1D

    dog_args: if full 2d DOGSpatFiltArgs then uses DOGSpatFiltArgs.to_dog_1d() to make 1d
    This takes the horizontal sd values

    collapse_symmetry: (default True)
        whether to multiply non-zero amplitudes by 2 to remove
        the positive/negative freq symmetry of fourier transforms
        True for sinusoidal amplitudes, False for convolutional amplitudes.

        Default is False, as the purpose of this func is direct calculation
        of convolutional amplitudes (or results of other extraneous processes)

    """

    if isinstance(dog_args, do.DOGSpatFiltArgs):
        dog_args_1d: do.DOGSpatFiltArgs1D = dog_args.to_dog_1d()
    else:
        dog_args_1d: do.DOGSpatFiltArgs1D = dog_args

    cent_ft = mk_gauss_1d_ft(
        freqs, amplitude=dog_args_1d.cent.amplitude, sd=dog_args_1d.cent.sd,
        collapse_symmetry=collapse_symmetry)
    surr_ft = mk_gauss_1d_ft(
        freqs, amplitude=dog_args_1d.surr.amplitude, sd=dog_args_1d.surr.sd,
        collapse_symmetry=collapse_symmetry)

    dog_ft_1d = cent_ft - surr_ft

    return dog_ft_1d


@overload
def mk_sf_ft_polar_freqs(
            theta: ArcLength[float],
            freq: SpatFrequency[float]
            ) -> Tuple[SpatFrequency[float], SpatFrequency[float]]: ...

@overload
def mk_sf_ft_polar_freqs(
            theta: ArcLength[np.ndarray],
            freq: SpatFrequency[float]
            ) -> Tuple[SpatFrequency[np.ndarray], SpatFrequency[np.ndarray]]: ...

@overload
def mk_sf_ft_polar_freqs(
            theta: ArcLength[float],
            freq: SpatFrequency[np.ndarray]
            ) -> Tuple[SpatFrequency[np.ndarray], SpatFrequency[np.ndarray]]: ...

def mk_sf_ft_polar_freqs(
            theta: Union[ArcLength[float], ArcLength[np.ndarray]],
            freq: Union[SpatFrequency[float], SpatFrequency[np.ndarray]]
            ) -> Tuple[SpatFrequency, SpatFrequency]:
    """Return 2D x and y freq values equivalent to polar args provided

    To aid in use of ft functions like [mk_dog_sf_ft][mk_dog_sf_ft] which require both
    x and y frequencies.

    Args:
        theta:
            * angle in the `2D` `FT` image
            * Also the angle of the **direction of modulation** of a sinusoid
            the in space domain (see notes).
            * EG, `0degs` --> **horizontal** modulation ---> **vertically** alligned grating

        freq: Actual frequency
            Magnitude of polar coordinate from center
            In 2D FFT, polar magnitude is frequency

    Returns:
        freq_x: cartesian freqs for 2D FFT along the `x` axis
        freq_y: cartesian freqs for 2D FFT along the `y` axis

    Notes:
        * `X frequencies` `-->` `vertically oriented gratings`
        * If a DOG rf is longer in the vertical than horizontal, and so
        "prefers" the "vertical" or `90deg` orientation, then it will have
        a higher preferred SF along the `x-axis` of frequencies, **not the y!**.
        * This is because a frequency is concerned with the **DIRECTION** in which
        a 2D sinusoidal is modulating.
        * Thus, a vertically aligned 2D sinusoidal grating actually modulates
        horizontally or along the `x` axis.

        ```
        --------  <---- * grating modulating "horizontally" or along x-axis
        --------        * theta=ArcLength(0, 'deg')
        --------        * freq_x=SpatFreq(2, 'cpd'), freq_y=SpatFreq(0, 'cpd')
        --------

        | | | |  <---- * grating modulating "vertically" or along y-axis
        | | | |        * theta=ArcLength(90, 'deg')
        | | | |        * freq_x=SpatFreq(0, 'cpd'), freq_y=SpatFreq(2, 'cpd')
        | | | |
        ```
    """
    # freq_x: SpatFrequency[val_gen]
    # freq_y: SpatFrequency[val_gen]
    freq_x = SpatFrequency(np.cos(theta.rad) * freq.cpd, 'cpd')
    freq_y = SpatFrequency(np.sin(theta.rad) * freq.cpd, 'cpd')

    return freq_x, freq_y


def mk_dog_sf_conv_amp(
        freqs_x: SpatFrequency[val_gen], freqs_y: SpatFrequency[val_gen],
        dog_args: do.DOGSpatFiltArgs,
        spat_res: ArcLength[float]
        ) -> val_gen:
    """Amplitude of result of convolving with dog_sf

    If convolving a sinusoid of a given frequency with this dog_sf filter
    as defined by the provided parameters, this will return the amplitude
    of the resultant sinusoid if the input sinusoid has amplitude 1

    Essentially same as mk_dog_sf_ft but divided by spat_res

    Parameters
    ----
    freqs_x, freqs_y:
        X and Y freq coordinates for the 2D Fourier Transform
        Can be derived with mk_sf_ft_polar_freqs
    dog_args:
        Definition of dog spatial filter
    spat_res:
        Spatial resolution of the stimulus that the spatial filter will
        be convolved with

    Returns
    ----
    amplitude as float/ndarray

    """

    dog_sf_amp = mk_dog_sf_ft(freqs_x=freqs_x, freqs_y=freqs_y, dog_args=dog_args)

    # return dog_sf_amp / spat_res.deg  # NOPE ... not degrees

    # >! Using a unit convention here!
    # use mnt as convention at the moment, and square as spatial
    return dog_sf_amp / (spat_res.mnt**2)


# > Orientation Biases

# functions for giving radially symmetric spatial filters an orientation bias
# by adjusting their horizontal and vertical SD values according to definitions
# from the literature

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
        r = mk_dog_sf_ft(x_freq, y_freq, sf_params)
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
    spat_freq_resp_v = mk_dog_sf_ft(
            *mk_sf_ft_polar_freqs(ArcLength(0), spat_freqs),
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

    sf_x, sf_y = mk_sf_ft_polar_freqs(angles, threshold_spat_freq)
    resp = mk_dog_sf_ft(sf_x, sf_y, new_sf_params)

    circ_var = cvvm.circ_var(resp, angles)

    return circ_var


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


circ_var_sd_ratio_methods = _CircVarSDRatioMethods()


# >> Create adjusted spatfilt parameters

def mk_ori_biased_spatfilt_params_from_spat_filt(
        spat_filt: do.DOGSpatialFilter,
        circ_var: float
        ) -> do.DOGSpatFiltArgs:
    """Create new spatial filter params from a spat filter by adding an orientation bias

    The bias will be for horizontal orientations (ie spat filt is stretched horizontally)

    Args:
        circ_var: the circular variance that the new parameters will have

    Returns:
        new params, with adjusted `sd` values so that the spatial filter
            they describe will have the provided `circ_var` (circular variance)
    """

    sd_ratio = spat_filt.ori_bias_params.circ_var2ratio(circ_var)
    # as first return val is biggest, this makes horizontally  elongated
    # orientation 0 degs
    h_sd_fact, v_sd_fact = mk_ori_biased_sd_factors(sd_ratio)

    new_sf_params = spat_filt.parameters.mk_ori_biased_duplicate(
        h_sd_factor=h_sd_fact, v_sd_factor=v_sd_fact
        )

    return new_sf_params


# > Temporal

# >> tq temp filt

# this is a hidden function because temp filters are simpler than spatial
# thus, spatial don't need hidden as simpler 1D functions are used for fitting
# for temporal, to suit the array args needed for fitting, a separation
# of all the args is helpful

def _mk_tqtempfilt(
        t: Time[val_gen],
        tau: Time[float] = Time(16, 'ms'),
        w: float = 4 * 2 * np.pi,
        phi: float = 0.24) -> val_gen:
    r"""Generate single temp filter by modulating a negative exp with a cosine

    Parameters
    ----
    tau : float (milliseconds)
        Time constant of negative exponential
    w : float (n*2*pi radians)
        Frequency of modulation of negative exp
        As in "n*2*pi" units, so that frequency is in Hertz
    phi : float (0.24)
        Phase translation of modulation

    Returns
    ----
    tf : (temporal filter) (array 1D)
        Magnitudes of the filter over time for each time step defined by
        the parameters

    Notes
    ----
    Taken from Teich and Qian (2006) and Chen et al (2001)

    On time constant and decay/growth dynamics for this filter, looking only at the
    gamma function or exponential component, the integral is:

    ```math
    F(t) = \int \frac{t \cdot exp(\frac{-t}{\tau})}{\tau^2} dt =
    -\frac{exp(\frac{-t}{\tau})(\tau + t)}{\tau}
    ```

    The definite integral from zero to a number of time constants \(n\tau\) is:

    ```math
    \begin{aligned}
        F(t)|^{n\tau}_{0} &= -\frac{\tau+n\tau}{\tau}exp(\frac{-n\tau}{\tau})
        - (-)\frac{\tau}{\tau}\exp(\frac{0}{\tau}) \\
                                            &= -(1+n)exp(-n) + 1 \\
                                            &= 1 - \frac{1+n}{e^n} \\
                                            &= 1 - \frac{1}{e^n} - \frac{n}{e^n}
    \end{aligned}
    ```

    Here, $` (1 - \frac{1}{e^n}) `$ is ordinary exponential growth/decay.
    The additional term of \(- \frac{n}{e^n}\) slows the growth/decay, but converges
    to zero by approx. 10 time constants (0.05%)
    """

    # if temp_ext_n_tau:

    #     assert temp_ext_n_tau >= 10, (
    #         f'temp_ext_n_tau ({temp_ext_n_tau}) should be 10 or more\n'
    #         f'Any lower and the sum of the filter will be diminished by approx 0.1% or more')
    #     temp_ext = tau * temp_ext_n_tau

    # converting to seconds from milliseconds
    # parameters of teich and qian function are in seconds
    # t = np.arange(0, temp_ext, temp_res) / 1000
    # tau /= 1000

    # Note correction from teich and qian (apparent in Kuhlman as well as Chen(?) too)
    # type ignores ... something wrong with ufuncs??
    exp_term: val_gen = np.exp(-t.s / tau.s)  # type: ignore
    cos_term: val_gen = np.cos((w * t.s) + phi)  # type: ignore

    tf = (t.s / tau.s**2) * exp_term * cos_term

    return tf


def mk_tq_tf(
        t: Time[val_gen],
        tf_params: do.TQTempFiltParams) -> val_gen:
    r"""Generate single temp filter by modulating a negative exp with a cosine

    Args passed to _mk_tqtempfilt

    Args:
        t: time
        tf_params: data object with amp, tau, w, phi


    Returns:
        tf: Magnitudes of the filter over time for each time step defined by
            the parameters

    Notes:

        Taken from Teich and Qian (2006) and Chen et al (2001)

        On time constant and decay/growth dynamics for this filter, looking only at the
        gamma function or exponential component, the integral is:

        ```math
        F(t) = \int \frac{t \cdot exp(\frac{-t}{\tau})}{\tau^2} dt =
        -\frac{exp(\frac{-t}{\tau})(\tau + t)}{\tau}
        ```

        The definite integral from zero to a number of time constants \(n\tau\) is:

        ```math
        \begin{aligned}
            F(t)|^{n\tau}_{0} &= -\frac{\tau+n\tau}{\tau}exp(\frac{-n\tau}{\tau})
            - (-)\frac{\tau}{\tau}\exp(\frac{0}{\tau}) \\
                                                &= -(1+n)exp(-n) + 1 \\
                                                &= 1 - \frac{1+n}{e^n} \\
                                                &= 1 - \frac{1}{e^n} - \frac{n}{e^n}
        \end{aligned}
        ```

        Here, $`1 - \frac{1}{e^n}`$ is ordinary exponential growth/decay.
        The additional term of $`- \frac{n}{e^n}`$ slows the growth/decay, but converges
        to zero by approx. 10 time constants (0.05%)
    """

    tf = tf_params.amplitude * _mk_tqtempfilt(
                    t,
                    tau=tf_params.arguments.tau,
                    w=tf_params.arguments.w,
                    phi=tf_params.arguments.phi
                )

    return tf


def _mk_tqtempfilt_ft(
        f: TempFrequency[val_gen],
        tau: Time[float] = Time(16, 'ms'),
        w: float = 4 * 2 * np.pi,
        phi: float = 0.24,
        return_abs: bool = True) -> val_gen:
    """Cerate Fourier Transform of function used in mk_tqtempfilt

    Employs analytical solution from Chen (2001)

    Args:
        f: Frequencies to calculate magnitude of in Fourier Transform Used in cycs per radian
        tau: as in mk_tqtempfilt. `tau` is transformed to seconds but taken in milliseconds
        w: as in mk_tqtempfilt
        phi: as in mk_tqtempfilt
        return_abs: Whether to return the absolute of the complex fourier transform array.
            This represents the magnitude of the fourier, and so is True be default

    Returns:
        fourier transform:

    Notes:
        For the magnitude of this fourier transform to correspond to that of a DFT (such as that
        computed by a FFT) you must divide it by the temporal period or sampling interval of the
        original filter or signal (ie, the tq filt being used in calculating the DFT)

        $`DFT(FFT) \\Leftrightarrow \\frac{FT_{continuous}}{T}`$

        This magnitude will also correspond with the result of convolving a sinusoid of the
        same frequency with the same original filter or signal.  IE, the magnitude of that
        frequency after filtering by this filter.

        Does not take a discrete data object as used in fitting and optmisation, which
        requires taking an array of values.
    """

    # tau /= 1000

    a: val_gen
    b: val_gen

    a = np.exp(phi * 1j) / (1 / tau.s + ((f.w - w) * 1j))**2  # type: ignore
    b = np.exp(-phi * 1j) / (1 / tau.s + ((f.w + w) * 1j))**2  # type: ignore

    fourier: val_gen
    fourier = (1 / (2 * tau.s**2)) * (a + b)  # type: ignore
    if return_abs:
        fourier: val_gen = np.abs(fourier)  # type: ignore

    return fourier


def mk_tq_tf_ft(
        freqs: TempFrequency[val_gen],
        tf_params: do.TQTempFiltParams) -> val_gen:
    """Generates fourer of temporal filter generated by mk_tq_tf

    Arguments passed to _mk_tqtempfilt_ft
    Fourer is analytical

    Parameters
    ----
    freqs: frequencies to generate fourier for
    tf_params: parameters defined by data object

    Returns
    ----
    fourier of tq temp filt
    """

    tf_ft = tf_params.amplitude * _mk_tqtempfilt_ft(
                                        f=freqs,
                                        tau=tf_params.arguments.tau,
                                        w=tf_params.arguments.w,
                                        phi=tf_params.arguments.phi
                                    )

    return tf_ft


# > Convolution adjustment or correction

def mk_tq_tf_conv_amp(
        freqs: TempFrequency[val_gen],
        tf_params: do.TQTempFiltParams,
        temp_res: Time[float]) -> val_gen:
    """Amplitude of convolution with given filter

    If convolving a sinusoid of a given frequency with this tq temp filter
    as defined by the provided parameters, this will return the amplitude
    of the resultant sinusoid if the input sinusoid has amplitude 1

    Essentially same as mk_tq_tf_ft but divided by temp_res


    Parameters
    ----


    Returns
    ----

    """

    tf_amp = mk_tq_tf_ft(freqs, tf_params)

    return tf_amp / temp_res.s  # use s as this unit used by mk_tq_tf (needs to be made cleaner!!)


# >> Joining Separable Spat and Temp

def joint_spat_temp_conv_amp(
        temp_freqs: TempFrequency[val_gen],
        spat_freqs_x: SpatFrequency[val_gen], spat_freqs_y: SpatFrequency[val_gen],
        tf: do.TQTempFilter, sf: do.DOGSpatialFilter, collapse_symmetry: bool = False
        ) -> val_gen:
    """Joint amplitude of separate TF and SF treated as separable

    Presumes TF and SF are sparable components of a single Spatia-Temporal
    filter.  TF and SF are defined by the tf and sf parameters.

    temp_res and spat_res represent the stimulus used for convolution.

    Parameters
    ----


    Returns
    ----

    """

    # sf of temp_filt
    tf_sf = tf.source_data.resp_params.sf
    sf_tf = sf.source_data.resp_params.tf

    norm_tf = mk_tq_tf_ft(sf_tf, tf.parameters)
    norm_sf = mk_dog_sf_ft(tf_sf, SpatFrequency(0), sf.parameters)
    # The amplitude that is what all amps are normlised to
    # norm_factor will normalise all amps to 1
    # this 1 will represent norm_amp which is the average of the spat and temp
    # responses
    norm_amp = (norm_tf + norm_sf) / 2
    norm_factor = norm_tf * norm_sf

    joint_amp: val_gen = (  # type: ignore
        mk_tq_tf_ft(temp_freqs, tf.parameters) *
        mk_dog_sf_ft(
            spat_freqs_x, spat_freqs_y, sf.parameters,
            collapse_symmetry=collapse_symmetry) /
        norm_factor *  # normalise to intersection
        norm_amp  # bring to normalised amplitude - avg of intersection
        )

    return joint_amp


def joint_dc(tf: do.TQTempFilter, sf: do.DOGSpatialFilter) -> float:
    """Calculate joint DC of temp and spat filters by averaging
    (must be within 30% of eachother)

    """

    # Joint DC Amplitude
    tf_dc = tf.source_data.resp_params.dc
    sf_dc = sf.source_data.resp_params.dc

    if abs(tf_dc - sf_dc) > 0.3*min([tf_dc, sf_dc]):
        tf_desc = (
            tf.source_data.meta_data.make_key()
            if tf.source_data.meta_data is not None
            else tf.parameters)
        sf_desc = (
            sf.source_data.meta_data.make_key()
            if sf.source_data.meta_data is not None
            else sf.parameters)
        raise ValueError(
            f'DC amplitudes of Temp Filt and Spat Filt are too differente\n'
            f'filters: {tf_desc}, {sf_desc}'
            f'DC amps: TF: {tf_dc}, SF: {sf_dc}'
            )
    # Just take the average
    joint_dc = (tf_dc + sf_dc) / 2

    return joint_dc


def mk_joint_sf_tf_resp_params(
        grating_stim_params: do.GratingStimulusParams,
        sf: do.DOGSpatialFilter, tf: do.TQTempFilter
        ) -> do.JointSpatTempResp:

    amplitude = joint_spat_temp_conv_amp(
        spat_freqs_x=grating_stim_params.spat_freq_x,
        spat_freqs_y=grating_stim_params.spat_freq_y,
        temp_freqs=grating_stim_params.temp_freq,
        sf=sf, tf=tf
        )

    DC = joint_dc(tf, sf)

    resp_estimate = do.JointSpatTempResp(amplitude, DC)

    return resp_estimate


def mk_estimate_sf_tf_conv_params(
        spacetime_params: do.SpaceTimeParams,
        grating_stim_params: do.GratingStimulusParams,
        sf: do.DOGSpatialFilter,
        tf: do.TQTempFilter
        ) -> do.EstSpatTempConvResp:
    """Produce estimate/analytical amplitude of response after convolving stim with tf and sf

    Presumes that convolving both a spatial (sf) and temporal (tf) filter with a grating
    grating stimulus defined by grating_stim_params
    """

    # should be convolution_amplitude after full sf and tf convolution with stim!!
    sf_conv_amp = mk_dog_sf_conv_amp(
        freqs_x=grating_stim_params.spat_freq_x,
        freqs_y=grating_stim_params.spat_freq_y,
        dog_args=sf.parameters, spat_res=spacetime_params.spat_res
        )
    tf_conv_amp = mk_tq_tf_conv_amp(
        freqs=grating_stim_params.temp_freq,
        temp_res=spacetime_params.temp_res,
        tf_params=tf.parameters
        )

    convolution_amp = grating_stim_params.amplitude * sf_conv_amp * tf_conv_amp

    sf_dc = mk_dog_sf_conv_amp(
                SpatFrequency(0), SpatFrequency(0), sf.parameters, spacetime_params.spat_res
                )
    tf_dc = mk_tq_tf_conv_amp(TempFrequency(0), tf.parameters, spacetime_params.temp_res)

    estimated_dc = grating_stim_params.amplitude * sf_dc * tf_dc

    conv_params = do.EstSpatTempConvResp(
        amplitude=convolution_amp,
        DC=estimated_dc
        )

    return conv_params


def mk_conv_resp_adjustment_params(
        spacetime_params: SpaceTimeParams,
        grating_stim_params: do.GratingStimulusParams,
        sf: do.DOGSpatialFilter,
        tf: do.TQTempFilter
        ) -> ConvRespAdjParams:
    """Factor to adjust amplitude of convolution to what filters dictate

    joint_spat_temp_conv_amp used to unify sf and tf
    stim attributes used to find appropriate factor for the expected output
    of convolution.

    Returned amplitude also adjusted for rectification effects.

    Args:
        spacetime_params: Parameters about space and time resolution and extent


    Returns:
        params: Adjustment parameters


    **Notes**

    Steps:

    1. Derive the amplitude of response of convolving TF.SF with stimulus.
        This is based on the parameters of the TF and SF and the amp of the stim.
    2. Derive the theoretical amplitude of a cell with the TF and SF at the freqs of the stimulus
    3. Derive the amplitude of an unrectified sin wave that would produce the above theoretical
        amplitude when rectified
    4. Produce factor that will normalise convolved amplitude to 1 and multiply by
        real unrectified amplitude necessary to produce theoretical.

    Thus, the factor returned is the theoretical amplitude / convolutional amplitude.

    """

    # should be convolution_amplitude after full sf and tf convolution with stim!!
    conv_resp_params = mk_estimate_sf_tf_conv_params(
        spacetime_params, grating_stim_params, sf, tf)

    # Joint response
    joint_resp_params = mk_joint_sf_tf_resp_params(grating_stim_params, sf, tf)

    # derive real unrectified amplitude
    real_unrect_joint_amp_opt = est_amp.find_real_f1(
        DC_amp=joint_resp_params.DC, f1_target=joint_resp_params.ampitude)

    real_unrect_joint_amp = real_unrect_joint_amp_opt.x[0]

    # factor to adjust convolution result by to provide realistic amplitude and firing rate
    # after convolution with a stimulus
    amp_adjustment_factor = real_unrect_joint_amp / conv_resp_params.amplitude

    # have to shift conv_resp estimate to same scale as joint_resp
    # means that must apply DC shift after (re-)scaling the response
    dc_shift_factor = joint_resp_params.DC - (conv_resp_params.DC * amp_adjustment_factor)

    adjustment_params = do.ConvRespAdjParams(
        amplitude=amp_adjustment_factor,
        DC=dc_shift_factor
        )

    return adjustment_params


def adjust_conv_resp(conv_resp: val_gen, adjustment: do.ConvRespAdjParams) -> val_gen:

    adjusted_resp = (conv_resp * adjustment.amplitude) + adjustment.DC

    return adjusted_resp


# > Temp Old Gauss (worgoter & Koch)

def mk_tempfilt(tau=10, temp_ext=100, temp_res=1, temp_ext_n_tau=None, return_t=False):
    '''Generate a temporal filter

    All time parameters in milliseconds

    Parameters
    ----
    tau : float
        Time constant
    temp_ext : float
        Length of temporal filter
    temp_res : float
        resolution of temporal filter
        size of temporal increment in milliseconds
    temp_ext_n_tau : int (Default None)
        length of temporl filter but in number of tau (time constants)
        If None, temp_ext used instead, if int, this is used
    return_t : Boolean
        If True, return time units array as well as magnitude of filer
        Else, return just the filter

    Maths and values taken from worgotter and koch (1991)
    '''

    # Use number of time consts as measure of temp extend, if specified
    if temp_ext_n_tau:
        temp_ext = temp_ext_n_tau * tau

    t = np.arange(0, temp_ext, temp_res)
    tf = np.exp(-t / tau) / tau

    if return_t:
        return t, tf
    else:
        return tf


def mk_tempfilt2(
        tau1=10, tau2=20,
        temp_res=1, temp_ext_n_tau=5, return_t=True,
        correct_integral_errors=True):
    """Generate two temporal filters with different time constants

    As with mk_tempfilt but with two taus


    Parameters
    ----
    correct_integral_errors : Boolean
        Ensure that both temporal filters sum to 1
        Necessary due to discretisation errors and need
        to ensure both filters do not alter the magnitude of the
        input signal.
        Done by dividing the temporal filter by the sum of the filter

    Returns
    ----
    tf1, tf2 : array (1D)
        EAch as the output of mk_tempfilt
        Magnitude of the filter at each time step in 1D filter
    """

    temp_ext = max([tau1, tau2]) * temp_ext_n_tau
    t, tf1 = mk_tempfilt(tau1, temp_ext, temp_res=temp_res, return_t=True)
    tf2 = mk_tempfilt(tau2, temp_ext, temp_res=temp_res)  # type: ignore
    tf1: np.ndarray
    tf2: np.ndarray

    if correct_integral_errors:
        tf1 /= tf1.sum()
        tf2 /= tf2.sum()

    if return_t:
        return t, tf1, tf2
    else:
        return tf1, tf2

