"""
Classes for handling and grouping basic data objects
"""

# # Imports

from __future__ import annotations
from functools import partial
from typing import (
    Union, Optional,
    Iterable, Iterator, Sequence, Dict, Any, Tuple, List, Literal, Set,
    overload, cast,
    Callable, Protocol
    )
from dataclasses import dataclass, astuple, asdict, field, replace, is_dataclass
from textwrap import dedent
import datetime as dt
from pathlib import Path
import pickle as pkl
import copy
import datetime as dt

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.interpolate import interp1d
import scipy.stats as stats
import pandas as pd

from . import settings, exceptions as exc
from .units.units import (
    val_gen, scalar,
    ArcLength, TempFrequency, SpatFrequency, Time
    )

import brian2 as bn
from brian2 import units as bnun


# ## Circular imports

# Likely to be circular, as we're using modules here that also use this one
# which isn't surprising as this is a central module, so any module outside
# of "utils" is likely to be circular if imported here.

# To avoid ... make functions that are "proxies" for an import but only execute
# "deferred imports"

def _filter_functions_module():
    import lif.receptive_field.filters.filter_functions as ff
    return ff

def _cv_von_mises_module():
    import lif.receptive_field.filters.cv_von_mises as cvvm
    return cvvm

# Old previous attempts at these imports ...

# try:
#     import lif.receptive_field.filters.filter_functions as ff
# except (ImportError, AttributeError):
#     import sys
#     root_package = __package__.split('.')[0]
#     ff = sys.modules[root_package + '.receptive_field.filters.filter_functions']
# try:
#     import lif.receptive_field.filters.cv_von_mises as cvvm
# except (ImportError, AttributeError):
#     import sys
#     root_package = __package__.split('.')[0]
#     cvvm = sys.modules[root_package + '.receptive_field.filters.cv_von_mises']

# import lif.receptive_field.filters.filter_functions as ff
# import lif.receptive_field.filters.cv_von_mises as cvvm

# from ..receptive_field.filters import (
#     filter_functions as ff,
#     cv_von_mises as cvvm
#     )


# numerical_iter = Union[np.ndarray, Iterable[float]]

PI: float = np.pi  # type: ignore

def is_dataclass_instance(obj):

    return (is_dataclass(obj) and not isinstance(obj, type))


class ConversionABC:
    """ABC for adding as dict and tuple methods from dataclasses module"""

    def asdict_(self) -> Dict[str, Any]:
        # intended to always used on data classes ... so just ignore here
        if not is_dataclass_instance(self):
            raise ValueError('Object is not a dataclass instance')
        else:
            return asdict(self)  # type: ignore

    def astuple_(self) -> Tuple[Any, ...]:
        # intended to always used on data classes ... so just ignore here
        if not is_dataclass_instance(self):
            raise ValueError('Object is not a dataclass instance')
        else:
            return astuple(self)  # type: ignore


@dataclass
class CitationMetaData(ConversionABC):
    author: str
    # """Author etc for source of data for a temp_filt fit"""
    year: int
    title: str
    reference: str
    doi: Optional[str] = None

    def _set_dt_uid(self, reset: bool = False) -> None:
        """Set a datetime based unique ID"""

        if hasattr(self, '_dt_uid') and not reset:
            pass
        else:
            self._dt_uid = dt.datetime.now()

    def make_key(self) -> str:
        if not (self.author or self.title):  # have neither author/title
            self._set_dt_uid()
            key = self._dt_uid.strftime('%y%m%d-%H%M%S')  # type: ignore 
        else:
            key_meta_data = {
                k: v for k, v in asdict(self).items() 
                if v is not None and k != 'doi'
                }
            key = '_'.join(
                    str(value).replace(' ', '_') 
                    for value in key_meta_data.values()
                )

        return key


# # Basic Optimisation Data object

# full optimisation result object is too much
# instead just capture the core data if I happen to want to need it later

@dataclass
class BasicOptimisationData(ConversionABC):
    """Simple stand in for a full OptimizeResult object with just the basic data
    """
    success: Optional[bool]
    cost: Optional[float]
    "Half the sum of squares of residuals (fun data)"
    x: Optional[np.ndarray]
    fun: Optional[np.ndarray]

    @classmethod
    def from_optimisation_result(
            cls, opt_result: OptimizeResult
            ) -> BasicOptimisationData:

        object_args = {
            # don't care too much about this data, Nones are fine if absent
            attribute: opt_result.get(attribute)
            for attribute in ("success", "cost", "x", "fun")
        }
        new_obj = cls(**object_args)

        return new_obj



# # Temp Filter

@dataclass
class TFRespMetaData(ConversionABC):
    """Metadata or additional data on context of temp filter data"""
    dc: Optional[float]
    sf: SpatFrequency[float]
    mean_lum: float
    contrast: float
    # Ideally would be ContrastValue, but that would break older data files

    def resolve(self):
        '''Fill missing values by some guided statistical means, return new object

        For new dc value where none is provided, exponential distribution used with
        parameters derived as follows:

        ```python
        def basic_exp(x, b):
            return np.exp(-b * x)

        # data from Kaplan_et_al_1987 fig 1
        xdata = np.array([10, 20, 30, 40])
        ydata = np.array([29, 17, 6, 1])
        ydata_norm = ydata / ydata.sum()

        opt_res = opt.curve_fit(
            basic_exp, xdata, ydata_norm, p0=[0.01]
            )

        opt_res[0]
        # 0.0638429
        ```
        '''
        # if DC provided, no need to generate new value
        if self.dc:
            return self

        b = 0.0638429  # hard coded from optimisation run (see docstring)

        new_dc = stats.expon.rvs(scale=1/b, size=1)[0]  # type: ignore
        new_dc = cast(float, new_dc)
        new_meta_data = copy.deepcopy(self)
        new_meta_data.dc = new_dc

        return new_meta_data

@dataclass
class TempFiltData(ConversionABC):
    """Temporal Frequency Response data

    frequencies: iter
    amplitudes: iter
    """
    frequencies: TempFrequency[np.ndarray]
    amplitudes: np.ndarray


@dataclass
class TempFiltParams(ConversionABC):
    """
    All data and information on temporal frequency response from literature
    """
    data: TempFiltData
    resp_params: TFRespMetaData
    meta_data: CitationMetaData


@dataclass
class TQTempFiltArgs(ConversionABC):
    """fundamental args for a tq filter, without arbitrary amplitude

    These describe the "shape" only

    tau, w, phi
    """
    tau: Time[float]
    w: float
    phi: float

    def array(self) -> np.ndarray:
        """
        tau.value, w, phi
        tau.value will be in original units
        """
        return np.array([self.tau.value, self.w, self.phi])
        # return np.array(astuple(self))


@dataclass
class TQTempFiltParams(ConversionABC):
    """Full parameters for a tq, including the amplitude

    For use with fitting to any real data, as arbitrary amplitude necessary for this

    amplitude: float
    arguments: TQTempFilter object
    """
    amplitude: float
    arguments: TQTempFiltArgs

    def array(self) -> np.ndarray:
        return np.r_[self.amplitude, self.arguments.array()]  # type: ignore

    @classmethod
    def from_iter(cls, data: Iterable[float], tau_time_unit: str = 's') -> TQTempFiltParams:
        """Create object from iterable

        iterable extects args: a, tau, w, phi

        tau_time_unit: unit for tau Time object
        """
        # IMPORTANT ... unpacking must match order above and in
        # definition of dataclasses!
        a, tau, w, phi = data

        return cls(
            amplitude=a,
            arguments=TQTempFiltArgs(
                tau=Time(tau, tau_time_unit), w=w, phi=phi
                )
            )

    def to_flat_dict(self) -> Dict[str, float]:
        """returns flat dictionary"""
        flat_dict = asdict(self)

        flat_dict.update(flat_dict['arguments'])  # add argument params
        del flat_dict['arguments']  # remove nested dict
        return flat_dict

    @property
    def parameters(self):
        """Convenience to allow either TQTempFilter or a TQTempFiltParams
        object to be used by a function whenever just using the TQTempFiltParams
        object (which is a child attribute of a TQTempFilter object) is
        desired.

        As TQTempFilter objects have TQTempFiltParams objects as "parameters",
        `tf_args = tf.parameters` will work in either case.
        Thus a variety of functions can happily take either kind of object.
        """
        return self


@dataclass
class TQTempFilter(ConversionABC):
    """Full tq temp filter that has been fit to data from the literature"""

    source_data: TempFiltParams
    """Data from which derived"""
    parameters: TQTempFiltParams
    optimisation_result: BasicOptimisationData

    @property
    def key(self) -> str:
        "Shortcut to getting key of filter"
        key: str = self.source_data.meta_data.make_key()

        return key

    def save(self, overwrite: bool = False):
        """Save this filter object to file as in pickle format.

        Will save to the data directory as defined in settings.

        Will not overwrite existing file unless arg `overwrite = True`
        """

        key: str = (
            self.source_data.meta_data.make_key()
            if self.source_data.meta_data is not None
            else 'Unknown'
            )
        file_name = Path(f'{key}-{self.__class__.__name__}.pkl')

        data_dir = settings.get_data_dir()
        if not data_dir.exists():
            data_dir.mkdir()

        data_file = data_dir / file_name

        if data_file.exists() and not overwrite:
            raise FileExistsError('Must passe overwrite=True to overwrite')

        with open(data_file, 'wb') as f:
            pkl.dump(self, f, protocol=4)

    @classmethod
    def get_saved_filters(cls) -> List[Path]:
        """Return list of temporal filters saved in data directory"""

        data_dir = settings.get_data_dir()
        pattern = f'*-{cls.__name__}.pkl'
        saved_filters = list(data_dir.glob(pattern))

        return saved_filters

    @staticmethod
    def load(path: Union[str, Path]) -> TQTempFilter:
        data_dir = settings.get_data_dir()
        data_path = data_dir / path

        if not data_path.exists() and data_path.is_file():
            raise FileNotFoundError(
                f'File {path} is not found in data dir {data_dir}')

        with open(data_path, 'rb') as f:
            temp_filt = pkl.load(f)

        return temp_filt



# As both have an equivalent "parameters" attribute, and so can be treated
# as the same functional type for certain purposes,
# this is a convenient type for such instances
TQTF = Union[TQTempFiltParams,TQTempFilter]
"Convenient type so that either may be provided though only parameters to be used"

# # Spatial Filter

@dataclass
class SpatFiltData(ConversionABC):

    amplitudes: np.ndarray
    frequencies: SpatFrequency[np.ndarray]


@dataclass
class SFRespMetaData(ConversionABC):
    """Metadata or additional data on context of spat filter data"""
    dc: Optional[float]
    "If not provided, `resolve()` method will generate one"
    tf: TempFrequency[float]  # temp frequency
    mean_lum: float
    # Ideally would be ContrastValue, but that'd break older saved data
    contrast: float

    def resolve(self):
        '''Fill missing values by some guided statistical means, return new object

        For new dc value where none is provided, exponential distribution used with
        parameters derived as follows:

        ```python
        def basic_exp(x, b):
            return np.exp(-b * x)

        # data from Kaplan_et_al_1987 fig 1
        xdata = np.array([10, 20, 30, 40])
        ydata = np.array([29, 17, 6, 1])
        ydata_norm = ydata / ydata.sum()

        opt_res = opt.curve_fit(
            basic_exp, xdata, ydata_norm, p0=[0.01]
            )

        opt_res[0]
        # 0.0638429
        ```
        '''
        # if DC provided, no need to generate new value
        if self.dc:
            return self

        b = 0.0638429  # hard coded from optimisation run (see docstring)

        new_dc = stats.expon.rvs(scale=1/b, size=1)[0]  # type: ignore
        new_dc = cast(float, new_dc)
        new_meta_data = copy.deepcopy(self)
        new_meta_data.dc = new_dc

        return new_meta_data


@dataclass
class SpatFiltParams(ConversionABC):
    """All data and information on spatial frequency response from the literature"""
    data: SpatFiltData
    resp_params: SFRespMetaData
    meta_data: CitationMetaData


@dataclass
class Gauss2DSpatFiltArgs(ConversionABC):
    """Args for shape of a 2d gaussian spatial filter

    h_sd: standard deviation along x-axis (horizontal)
    v_sd: standard deviation along y-axis (vertical)

    """
    h_sd: ArcLength[float]
    v_sd: ArcLength[float]

    def array(self) -> np.ndarray:
        args_array = np.array([self.h_sd.base, self.v_sd.base])
        return args_array

    def mk_ori_biased_duplicate(
            self,
            v_sd_factor: float, h_sd_factor: float
            ) -> Gauss2DSpatFiltArgs:

        new_v_sd = ArcLength(self.v_sd.value * v_sd_factor, self.v_sd.unit)
        new_h_sd = ArcLength(self.h_sd.value * h_sd_factor, self.h_sd.unit)

        return self.__class__(h_sd=new_h_sd, v_sd = new_v_sd)




@dataclass
class Gauss2DSpatFiltParams(ConversionABC):
    """Args for a 2d gaussian spatial filter

    amplitude: magnitude
    arguments: Gauss2DSpatFiltArgs (main shape arguments)

    """
    amplitude: float
    arguments: Gauss2DSpatFiltArgs

    def array(self) -> np.ndarray:
        return np.r_[self.amplitude, self.arguments.array()]

    @classmethod
    def from_iter(cls, data: Iterable[float], arclength_unit: str = 'deg') -> Gauss2DSpatFiltParams:
        """Create object from iterable (a, h_sd, v_sd)
        """
        # IMPORTANT ... unpacking must match order above and in
        # definition of dataclasses!
        a, h_sd, v_sd = data

        return cls(
            amplitude=a,
            arguments=Gauss2DSpatFiltArgs(
                h_sd=ArcLength(h_sd, arclength_unit), v_sd=ArcLength(v_sd, arclength_unit)
                ))

    def to_flat_dict(self) -> Dict[str, float]:
        """returns flat dictionary"""
        flat_dict = asdict(self)

        flat_dict.update(flat_dict['arguments'])  # add argument params
        del flat_dict['arguments']  # remove nested dict
        return flat_dict


@dataclass
class Gauss1DSpatFiltParams(ConversionABC):
    """Args for a 1D gaussian spatial filter

    amplitude: magnitude
    arguments: Gauss2DSpatFiltArgs (main shape arguments)
    """
    amplitude: float
    sd: ArcLength[float]

    def array(self) -> np.ndarray:
        return np.array([self.amplitude, self.sd.value])  # type: ignore

    @classmethod
    def from_iter(cls, data: Iterable[float], arclength_unit: str = 'deg') -> Gauss1DSpatFiltParams:
        """Create object from iterable... (a, h_sd, v_sd)
        """
        # IMPORTANT ... unpacking must match order above and in
        # definition of dataclasses!
        a, sd = data

        return cls(amplitude=a, sd=ArcLength(sd, arclength_unit))


@dataclass
class DOGSpatFiltArgs1D(ConversionABC):

    cent: Gauss1DSpatFiltParams
    surr: Gauss1DSpatFiltParams

    def array(self) -> np.ndarray:
        """cent args, surr args"""
        return np.hstack((self.cent.array(), self.surr.array()))

    @classmethod
    def from_iter(cls, data: Iterable[float], arclength_unit: str = 'deg') -> DOGSpatFiltArgs1D:
        """cent_a, cent_sd, surr_a, surr_sd"""
        cent_a, cent_sd, surr_a, surr_sd = data

        cent = Gauss1DSpatFiltParams(cent_a, ArcLength(cent_sd, arclength_unit))
        surr = Gauss1DSpatFiltParams(surr_a, ArcLength(surr_sd, arclength_unit))

        return cls(cent=cent, surr=surr)


@dataclass
class DOGSpatFiltArgs(ConversionABC):
    """Args for a DoG spatial filter... 2 2dGauss, one subtracted from the other
    """
    cent: Gauss2DSpatFiltParams
    surr: Gauss2DSpatFiltParams

    def array(self) -> np.ndarray:
        """1D array of [cent args, surr args]"""
        return np.hstack((self.cent.array(), self.surr.array()))

    def array2(self) -> np.ndarray:
        """2D array of [[cent args], [surr args]]"""
        return np.vstack((self.cent.array(), self.surr.array()))

    def max_sd(self) -> ArcLength[float]:
        """Max sd val in definition of parameters as ArcLength, unit preserved"""

        sds = (
            self.cent.arguments.h_sd,
            self.cent.arguments.v_sd,
            self.surr.arguments.h_sd,
            self.surr.arguments.v_sd
            )

        max_sd = max(sds, key=lambda sd: sd.base)

        return ArcLength(max_sd.value, max_sd.unit)


    def to_dog_1d(self) -> DOGSpatFiltArgs1D:
        """Take h_sd values, presume radial symmetry, return 1D DoG spat fil"""

        cent_a, cent_h_sd, _, surr_a, surr_h_sd, _ = self.array()
        return DOGSpatFiltArgs1D.from_iter([cent_a, cent_h_sd, surr_a, surr_h_sd])

    def mk_ori_biased_duplicate(
            self, v_sd_factor: float, h_sd_factor: float
            ) -> DOGSpatFiltArgs:
        """Create duplicate but with factors applied to sd values (with horizontal 0deg ori)"""

        spat_filt_args = copy.deepcopy(self)

        # alter only the center gaussian
        spat_filt_args.cent.arguments = (
            spat_filt_args.cent.arguments
            .mk_ori_biased_duplicate(v_sd_factor=v_sd_factor, h_sd_factor=h_sd_factor)
            )

        return spat_filt_args

    @property
    def parameters(self):
        """Convenience to allow either DOGSpatFiltArgs or a DOGSpatialFilter
        object to be used by a function whenever just using the DOGSpatFiltArgs
        object (which is a child attribute of a DOGSpatialFilter object) is
        desired.

        As DOGSpatialFilter objects have DOGSpatFiltArgs objects as "parameters",
        `sf_args = sf.parameters` will work in either case.
        Thus a variety of functions can happily take either kind of object.
        """
        return self

# ## Circular Variance Objects

@dataclass
class CircularVarianceSDRatioVals(ConversionABC):
    """Circular Variance to SD ration values for a particular method"""

    sd_ratio_vals: np.ndarray = field(repr=False)
    circular_variance_vals: np.ndarray = field(repr=False)
    method: str
    _max_sd_ratio: float = field(repr=False)
    "Highest sd ratio used to generate lookup tables"
    _max_circ_var: float = field(repr=False)
    "Highest circular variance used/generated in lookup tables"

    def ratio2circ_var(self, ratio: val_gen) -> val_gen:
        """Return circular variance for given sd ratio
        """

        if not hasattr(self, '_circ_var'):
            self._mk_interpolated()

        # bounds check ratio (not below 1)
        if np.any(ratio < 1):  # type: ignore
            raise ValueError(f'Ratio must be >= 1, is {ratio}')

        return self._circ_var(ratio)

    def circ_var2ratio(self, circ_var: val_gen) -> val_gen:
        """Return sd ratio for given circular variance
        """

        if not hasattr(self, '_sd_ratio'):
            self._mk_interpolated()

        # bounds circ_var ratio ( between 0 and 1 )
        if not (np.all(circ_var >= 0) and np.all(circ_var <= 1)):  # type: ignore
            raise ValueError(f'Circ var must be between 0 and 1, is {circ_var}')

        return self._sd_ratio(circ_var)

    def _mk_interpolated(self):
        """Add methods for interpolation in both directions (using interp1d)"""

        # bit clunky and hacky here, along with the checking for attrs above
        # this should be in a post_init method.
        interp_func = partial(interp1d, fill_value='extrapolate')

        self._sd_ratio = interp_func(
            self.circular_variance_vals, self.sd_ratio_vals)
        self._circ_var = interp_func(
            self.sd_ratio_vals, self.circular_variance_vals)

@dataclass
class CircularVarianceParams(ConversionABC):
    """Interface to obtaining SD Ratio values"""

    naito: CircularVarianceSDRatioVals
    shou: CircularVarianceSDRatioVals
    # leventhal: CircularVarianceSDRatioVals

    def get_method(self, method: str) -> CircularVarianceSDRatioVals:
        return self.__getattribute__(method)

    @classmethod
    def all_methods(cls) -> List[str]:
        "List of all avialble methods"
        return list(cls.__dataclass_fields__.keys())

    def ratio2circ_var(self, ratio: val_gen, method = 'naito') -> val_gen:
        lookup_obj = self.get_method(method)
        return lookup_obj.ratio2circ_var(ratio)

    def circ_var2ratio(self, circ_var: val_gen, method = 'naito') -> val_gen:
        lookup_obj = self.get_method(method)
        return lookup_obj.circ_var2ratio(circ_var)


@dataclass
class DOGSpatialFilter(ConversionABC):
    """Full DoG Spatial filter that has been fit to data from the literature"""

    source_data: SpatFiltParams
    parameters: DOGSpatFiltArgs
    """Arguments needed to generate the spatial filter in this code base"""
    optimisation_result: BasicOptimisationData
    """Output of the optimisation process/function"""
    # ori_bias_params: CircularVarianceSDRatioVals
    ori_bias_params: CircularVarianceParams
    """Values for generating orientation biased version of this filter"""

    @property
    def key(self) -> str:
        "Shortcut to getting key of filter"
        key: str = self.source_data.meta_data.make_key()

        return key

    def save(self, overwrite: bool = False, custom_key: str = 'Unknown'):

        key: str = (
            self.source_data.meta_data.make_key()
            if self.source_data.meta_data is not None
            else custom_key
            )
        file_name = Path(f'{key}-{self.__class__.__name__}.pkl')

        data_dir = settings.get_data_dir()
        if not data_dir.exists():
            data_dir.mkdir()

        data_file = data_dir / file_name

        if data_file.exists() and not overwrite:
            raise FileExistsError('Must passe overwrite=True to overwrite')

        with open(data_file, 'wb') as f:
            pkl.dump(self, f, protocol=4)

    @classmethod
    def get_saved_filters(cls) -> List[Path]:
        """Return list of spatial filters saved in data directory"""

        data_dir = settings.get_data_dir()
        pattern = f'*-{cls.__name__}.pkl'
        saved_filters = list(data_dir.glob(pattern))

        return saved_filters

    @staticmethod
    def load(path: Union[str, Path]) -> DOGSpatialFilter:
        data_dir = settings.get_data_dir()
        data_path = data_dir / path

        if not data_path.exists() and data_path.is_file():
            raise FileNotFoundError(
                f'File {path} is not found in data dir {data_dir}')

        with open(data_path, 'rb') as f:
            spat_filt = pkl.load(f)

        return spat_filt

    # def resolve(self):
    #     """Perform any arbitrary actions required to fill in any missing values

    #     This will return a new deeply copied object with the value replaced

    #     Current:
    #         Add a DC value to the repsonse params if there wasn't one provided
    #     """


# As both have an equivalent "parameters" attribute, and so can be treated
# as the same functional type for certain purposes,
# this is a convenient type for such instances
DOGSF = Union[DOGSpatialFilter,DOGSpatFiltArgs]
"Convenient type so that either may be provided though only parameters to be used"

# # Full LGN Model

@dataclass
class RFLocation(ConversionABC):
    """Spatial Location of a single receptive field

    Uses cartesian coords (in arc lengths), but
    has constructor using polar coords if necessary.
    """
    x: ArcLength[scalar]
    y: ArcLength[scalar]

    @classmethod
    def from_polar(
        cls,
        theta: ArcLength[scalar], mag: ArcLength[scalar],
        unit: str = 'mnt') -> 'RFLocation':
        """Construct using polar coordinates

        * `theta` and `mag` define a vector.
        * `unit` defines the `ArcLength` unit for the final `x`,`y` coords
        """
        x = ArcLength(mag[unit] * np.cos(theta.rad), unit)
        y = ArcLength(mag[unit] * np.sin(theta.rad), unit)

        return cls(x=x, y=y)

    def round_to_spat_res(self, spat_res: ArcLength[int]) -> 'RFLocation':
        """New object with `x`,`y` coords snapped to `spat_res`
        """
        ff=_filter_functions_module()
        x = ff.round_coord_to_res(coord=self.x, res=spat_res)
        y = ff.round_coord_to_res(coord=self.y, res=spat_res)

        return self.__class__(x=x, y=y)

    @property
    def arclength_unit(self) -> str:
        """Get the unit that the coords are in

        If inconsistent, raise an exception (CoordsValueError)
        """
        if self.x.unit != self.y.unit:
            raise exc.CoordsValueError(
                f'X and Y do not have the same unit: ({self.x.unit, self.y.unit})')
        else:
            return self.x.unit

@dataclass
class RFStimSpatIndices(ConversionABC):
    """Indices to use for slicing a RF's view of a stimulus

    Indices are to be used directly on the array

    Examples:
        # rf_idxs: RFStimIndices
        # Note, y goes first, as meshgrid by default puts Y as the first dimension
        >>> stim[rf_idxs.y1:rf_idxs.y2, rf_idxs.x1:rf_idxs.x2]
    """
    x1: int
    x2: int
    y1: int
    y2: int

    def is_within_extent(self, st_params: SpaceTimeParams) -> bool:
        """Do all indices fall within the shape of the stimulus array?

        Uses the space time parameters to determine what the extent will be,
        then checkes of all indices are within 0 and the predicted size.
        """
        ff = _filter_functions_module()
        full_ext = ff.spatial_extent_in_res_units(
            st_params.spat_res, spat_ext = st_params.spat_ext)

        return all(
            0 <= self.__getattribute__(idx) < full_ext
                for idx in ('x1', 'x2', 'y1', 'y2')
            )


# ## Location Distributions

# +
class BivariateGaussParams(ConversionABC):

    @overload
    def __init__(self, sigma_x: scalar, sigma_y: scalar, ratio: None = None, ): ...
    @overload
    def __init__(self, sigma_x: scalar, sigma_y: None = None, *, ratio: scalar): ...
    def __init__(self,
            sigma_x: scalar,
            sigma_y: Optional[scalar] = None,
            ratio: Optional[scalar] = None,
            ):
        """Bivariate Guassian for 2D cartesion coordinates of LGN inputs

        Unitless ... sigma values are unitless

        Examples:
            >>> t = BivariateGaussParams(sigma_x = 1.25, ratio=3)
            >>> t.sigma_y
            ... 3.75
        """

        if (
                (sigma_y is None) and (ratio is None)
                or
                (sigma_y and ratio)
            ):
                raise exc.LGNError('Must have only one of `sigma_y` or `ratio` value')
        if ratio:
            self.sigma_y = sigma_x * ratio
        elif isinstance(sigma_y, (int, float)):
            self.sigma_y = sigma_y

        self.sigma_y = cast(scalar, self.sigma_y)
        self.sigma_x: scalar = sigma_x
        self.ratio = ratio

    def axial_difference_sigma_vals(self) -> BivariateGaussParams:
        """Sigma values for the distribution of distances along x and y axes

        Uses the fact that gaussian variances sum linearly.
        Thus, variance for differences along the X axis is `2*sigma_x^2`.
        The std deviation (ie sq-root) is therefore `(2^0.5)*sigma_x`.
        Here a new object is returned with the sigma values scaled by root-2 to
        represent the distribution of axial differences.
        """
        return self.__class__(
                    sigma_x = (2**0.5)*self.sigma_x,
                    sigma_y = (2**0.5)*self.sigma_y
                    )

    def __repr__(self):
        return f'BivariateGaussParams(sigma_x={self.sigma_x}, sigma_y={self.sigma_y}{", ratio="+str(self.ratio) if self.ratio else ""})'

# -

# +
@dataclass
class RatioSigmaXOptLookUpVals(ConversionABC):
    ratios: np.ndarray
    sigma_x: np.ndarray
    errors: np.ndarray

    def to_df(self):
        df = pd.DataFrame({
            'ratios': self.ratios,
            'sigma_x': self.sigma_x,
            'error': self.errors
            })

        return df

@dataclass
class RFLocMetaData:
    source: str
    "Major source such as a paper"
    specific_src: str
    "Figure or table"
    comment: Optional[str] = None
    "If desired ... won't be part of file name"

    def mk_key(self):
        key = f'{self.source}-{self.specific_src}'
        return key

@dataclass
class RFLocationSigmaRatio2SigmaVals:
    """Optimum sigma values for bivariate gaussian rf loc with pairwise dists matching data

    Use `ratio2gauss_params` to convert a ratio (of x and y sigma values)
    to a `BivariateGaussParams` object

    For saving an object or loading from file use:
    * `save()`
    * `load()`
    * `get_saved_rf_loc_generators()` (lists files already saved with an object of this type)
    """
    meta_data: RFLocMetaData
    "Information on source of data the bivariate gaussian is fit to"
    lookup_vals: RatioSigmaXOptLookUpVals = field(repr=False)
    "Raw table of data used to lookup an optimal sigma value"
    data_bins: np.ndarray
    "Bins of the histogram of data that was optimised to"
    data_prob: np.ndarray
    "Values, as probabilities, of the data that was fit to"

    def ratio2gauss_params(self, ratio: float) -> BivariateGaussParams:
        """Provide bivariate sigma values optimised to provided ratio
        """
        if not hasattr(self, '_sigma_x'):
            self._mk_interpolated()

        sigma_x_val = self._sigma_x(ratio)
        gauss_params = BivariateGaussParams(sigma_x=sigma_x_val, ratio=ratio)
        return gauss_params

    def _mk_interpolated(self):
        """Add methods for interpolation"""
        self._sigma_x = interp1d(x=self.lookup_vals.ratios, y=self.lookup_vals.sigma_x)

    @classmethod
    @property
    def _filename_template(cls) -> Callable:
        """Returns format function of `"RfLoc_Generator_{}.pkl"`"""
        return 'RfLoc_Generator_{}.pkl'.format

    def _mk_filename(self) -> Path:
        "Filename for this object to be saved to and identifiable from"

        file_name = Path(self._filename_template(self.meta_data.mk_key()))
        return file_name

    def _mk_data_path(self) -> Path:
        "Use settings to retrieve the path for saving/loading data"
        data_dir = settings.get_data_dir()
        if not data_dir.exists():
            data_dir.mkdir()

        file_name = self._mk_filename()
        data_file = data_dir / file_name

        return data_file

    def save(self, overwrite: bool = False):
        "Save this object to file"

        data_file = self._mk_data_path()

        if data_file.exists():
            if not overwrite:
                raise FileExistsError('Must passe overwrite=True to overwrite')

        with open(data_file, 'wb') as f:
            try:
                pkl.dump(self, f, protocol=4)
            except Exception as e:
                # don't want bad file floating around
                # if overwrite ... well bad luck
                data_file.unlink()
                raise e

    def __getstate__(self):
        '''Don't want to save the interpolation functions, they are created automatically

        This dunder method interfaces with the pickle library.
        The object returned is what actually gets pickled
        '''
        interp_func_name = '_sigma_x'
        state = self.__dict__.copy()
        if interp_func_name in state:
            del state[interp_func_name]
        return state

    def __setstate__(self, state):
        '''How loading from pickle works ... here lets just recreate the interpolation'''
        self.__dict__.update(state)
        self._mk_interpolated()


    @classmethod
    def get_saved_rf_loc_generators(cls) -> List[Path]:
        """Return list of location generator objects saved in data directory"""

        data_dir = settings.get_data_dir()
        pattern = cls._filename_template('*')  # glob what is supposed to be the key
        saved_filters = list(data_dir.glob(pattern))

        return saved_filters

    @staticmethod
    def load(path: Union[str, Path]) -> RFLocationSigmaRatio2SigmaVals:
        "Load pickle from path"
        data_dir = settings.get_data_dir()
        data_path = data_dir / path

        if not (data_path.exists() and data_path.is_file()):
            raise FileNotFoundError(
                f'File {path} is not found in data dir {data_dir}')

        with open(data_path, 'rb') as f:
            rf_loc_generator = pkl.load(f)

        if not isinstance(rf_loc_generator, RFLocationSigmaRatio2SigmaVals):
            raise ValueError(
                f'Provided path ({path}) does not contain a RF Locations object of type {RFLocationSigmaRatio2SigmaVals}')

        return rf_loc_generator
# -

@dataclass
class RFLocationScalingCoefficiantLUV:
    ratios: np.ndarray
    coefficients: np.ndarray

@dataclass
class RFLocationScalingCoefficientVals:
    spat_filt_keys: Set[str]
    "The spatial filters for which these lookup vals are optimised"
    lookup_vals: RFLocationScalingCoefficiantLUV
    "Raw discrete look up vals to interpolate in providing a coefficient for a given ratio"
    meta_data: RFLocMetaData
    "Source of data to which lookup vals were fit"
    data_bins: np.ndarray
    "Bins of the histogram of data that was optimised to"
    data_prob: np.ndarray
    "Values, as probabilities, of the data that was fit to"

    def check_spat_filt_match(self, spat_filt_keys: Sequence[str]) -> bool:
        '''Check that the provided spatial filters match those for which these LUVs were optimised
        '''

        if self.spat_filt_keys == set(spat_filt_keys):
            return True
        else:
            return False

    def ratio2coefficient(self, ratio: float) -> float:
        """Provide bivariate sigma values optimised to provided ratio
        """
        if not hasattr(self, '_coefficient'):
            self._mk_interpolated()

        coefficient_val = self._coefficient(ratio)
        return coefficient_val

    def _mk_interpolated(self):
        """Add methods for interpolation"""
        self._coefficient = interp1d(x=self.lookup_vals.ratios, y=self.lookup_vals.coefficients)

    @classmethod
    @property
    def _filename_template(cls) -> Callable:
        """Returns format function of `"RfLoc_Generator_{}.pkl"`"""
        return 'RfLoc_scaling_coefficients_{}.pkl'.format

    def _mk_filename(self) -> Path:
        "Filename for this object to be saved to and identifiable from"

        file_name = Path(self._filename_template(self.meta_data.mk_key()))
        return file_name

    def _mk_data_path(self) -> Path:
        "Use settings to retrieve the path for saving/loading data"
        data_dir = settings.get_data_dir()
        if not data_dir.exists():
            data_dir.mkdir()

        file_name = self._mk_filename()
        data_file = data_dir / file_name

        return data_file

    def save(self, overwrite: bool = False):
        "Save this object to file"

        data_file = self._mk_data_path()

        if data_file.exists():
            if not overwrite:
                raise FileExistsError('Must passe overwrite=True to overwrite')

        with open(data_file, 'wb') as f:
            try:
                pkl.dump(self, f, protocol=4)
            except Exception as e:
                # don't want bad file floating around
                # if overwrite ... well bad luck
                data_file.unlink()
                raise e

    def __getstate__(self):
        '''Don't want to save the interpolation functions, they are created automatically

        This dunder method interfaces with the pickle library.
        The object returned is what actually gets pickled
        '''
        interp_func_name = '_coefficient'
        state = self.__dict__.copy()
        if interp_func_name in state:
            del state[interp_func_name]
        return state

    def __setstate__(self, state):
        '''How loading from pickle works ... here lets just recreate the interpolation'''
        self.__dict__.update(state)
        self._mk_interpolated()


    @classmethod
    def get_saved_rf_loc_generators(cls) -> List[Path]:
        """Return list of location generator objects saved in data directory"""

        data_dir = settings.get_data_dir()
        pattern = cls._filename_template('*')  # glob what is supposed to be the key
        saved_filters = list(data_dir.glob(pattern))

        return saved_filters

    @staticmethod
    def load(path: Union[str, Path]) -> RFLocationSigmaRatio2SigmaVals:
        "Load pickle from path"
        data_dir = settings.get_data_dir()
        data_path = data_dir / path

        if not (data_path.exists() and data_path.is_file()):
            raise FileNotFoundError(
                f'File {path} is not found in data dir {data_dir}')

        with open(data_path, 'rb') as f:
            rf_loc_generator = pkl.load(f)

        if not isinstance(rf_loc_generator, RFLocationSigmaRatio2SigmaVals):
            raise ValueError(
                f'Provided path ({path}) does not contain a RF Locations object of type {RFLocationSigmaRatio2SigmaVals}')

        return rf_loc_generator

# ## Orientations and Circular Variance

@dataclass
class VonMisesParams(ConversionABC):
    phi: ArcLength[scalar]
    k: float
    a: float = field(default=1, repr=False)

    @classmethod
    def from_circ_var(cls, cv: float, phi: ArcLength[scalar], a: float=1) -> VonMisesParams:

        cvvm = _cv_von_mises_module()
        return cls(
            k=cvvm.cv_k(cv),
            phi=phi,
            a=a)


# ### Circ Var distributions

@dataclass
class CircVarHistData:
    hist_mp: np.ndarray
    "mid point values of bins"
    count: np.ndarray
    _bins: Optional[np.ndarray] = None

    @property
    def hist_bins(self, width: Optional[float]=None, overwrite: bool = False) -> np.ndarray:
        "Bin boundaries from mid points"

        if (self._bins is not None) and (not overwrite):
            return self._bins

        if width is None:
            bin_diffs = np.diff(self.hist_mp)
            if not np.allclose(bin_diffs, bin_diffs[0]):
                raise ValueError('Width of Bin widths cannot be inferred as irregular')
            width = bin_diffs[0]
            width = cast(float, width)

        self._bins = np.r_[0, (self.hist_mp + (width/2))]
        return self._bins

    @property
    def probs(self) -> np.ndarray:
        return self.count / self.count.sum()


class DistProtocol(Protocol):
    """Basic type intended to represent a frozen rv from `scipy.stats`

    Just the basic methods used here
    """
    def pdf(self, x: val_gen) -> val_gen: ...
    def cdf(self, x: val_gen) -> val_gen: ...
    @overload
    def rvs(self, size: None = None) -> float: ...
    @overload
    def rvs(self, size: int) -> np.ndarray: ...
    def rvs(self, size: Optional[int] = None) -> Union[float, np.ndarray]: ...


@dataclass
class CircVarianceDistribution:
    """A single distribution instantiated form a single dataset
    """
    name: str
    source: str
    "Data from which distribution derived or fit to"
    specific: str
    "Specific reference within source such as figure number"
    distribution: DistProtocol
    "`scipy.stats` object that has been fit to the data"
    raw_data: CircVarHistData


@dataclass
class AllCircVarianceDistributions:
    "All distribution objects created in this module"
    naito_lg_highsf: CircVarianceDistribution
    naito_opt_highsf: CircVarianceDistribution
    naito_lg_optsf: CircVarianceDistribution
    shou_xcells: CircVarianceDistribution

    def get_distribution(self, alias: str) -> CircVarianceDistribution:

        try:
            distribution = self.__getattribute__(alias)
        except AttributeError as e:
            raise exc.LGNError(dedent(f'''
                Circular Variance Distribution {alias} not available.
                Available options from {self.__class__.__name__}:
                {self.__dataclass_fields__.keys()}
                '''
                )) from e

        distribution = cast(CircVarianceDistribution, distribution)

        return distribution


# ## Contrast Parameters

@dataclass(frozen=True)
class ContrastValue:
    contrast: scalar

    def __post_init__(self):

        if (0 >= self.contrast) or (self.contrast >= 1):
            raise ValueError(f'contrast must be between 0 and 1')


@dataclass(frozen=True)
class ContrastParams:
    """Parameters for a contrast response curve

    Intended to work with contrast curves of form:

    `R = (Rmax*C**(n))/(C_50**(n) + C**(n))`

    Notes:
        Taken from:

        * Cheng, H., Chino, Y. M., Smith, E. L., Hamamoto, J., & Yoshida, K. (1995).
        Transfer characteristics of lateral geniculate nucleus X neurons in the cat: Effects of spatial frequency and contrast.
        Journal of Neurophysiology, 74(6), 2548–2557.
        * Troyer, T. W., Krukowski, A. E., Priebe, N. J., & Miller, K. D. (1998).
        Contrast-Invariant Orientation Tuning in Cat Visual Cortex: Thalamocortical Input Tuning and Correlation-Based Intracortical Connectivity.
        Journal of Neuroscience, 18(15), 5908–5927.

    """
    max_resp: float
    "Maximum firing rate in Hz"
    contrast_50: float
    "contrast which elicits a response half of maximum"
    exponent: float
    "characterises the curve shape"

    def to_array(self) -> np.ndarray:
        return np.array([self.max_resp, self.contrast_50, self.exponent])

    @classmethod
    def from_array(cls, array) -> ContrastParams:

        return cls(max_resp=array[0], contrast_50=array[1], exponent=array[2])

    def adjusted_max_resp_to_target_values(
            self, contrast: float, resp: float
            ) -> ContrastParams:
        """Shift max resp so that target contrast and response value lie on curve

        Shifting done by deriving the new max_resp value from the formula algebraically:

        `Rmax = (R * (C_50**(n) + C**(n))) / C**(n)`
        """

        new_max_resp = (
            resp * (self.contrast_50**(self.exponent) + contrast**(self.exponent))
            /
            (contrast**(self.exponent))
            )

        return replace(self, max_resp = new_max_resp)


# ## Max Firing Rate (F1 Amp) Distribution

@dataclass
class LGNF1AmpMaxValue(ConversionABC):
    max_amp: float
    contrast: ContrastValue


@dataclass
class LGNActualF1AmpMax(ConversionABC):
    "For recording actual maximal amplitudes of actual filters"
    value: LGNF1AmpMaxValue
    temp_freq: TempFrequency
    "temporal frequency at which maximum occurs"
    spat_freq: SpatFrequency
    "spatial frequency at which maximum occurs"

@dataclass
class LGNF1AmpDistParams(ConversionABC):
    """Distribution for Max F1 amplitude of LGN cells

    Default values are fine ... Drawn, approximately, from Table 2 of Saul and Humphrey 1990

    Default parameters approximately the data for a modulating spot in table 2
    """
    alpha: float = 10
    loc: float = 24
    scale: float = 28
    contrast: ContrastValue = ContrastValue(0.4)
    "Contrast at which the represented data was generated"

    def draw_f1_amp_vals(self, n: int) -> Sequence[LGNF1AmpMaxValue]:

        f1_rvs: np.ndarray = (
                 stats
                 .skewnorm(
                    self.alpha,
                    loc=self.loc,
                    scale=self.scale)
                 .rvs(size=n)
                 )
        f1_max_vals = [
            LGNF1AmpMaxValue(max_amp=max_amp, contrast=self.contrast)
                for max_amp in f1_rvs
        ]

        return f1_max_vals


# ## Full LGN Cell

@dataclass
class LGNCellRecord(ConversionABC):
    """For storing an LGN Cell on disk, easily rendered as a full cell"""
    spat_filt: str
    "key to actual spatial filter file name"
    oriented_spat_filt_params: DOGSpatFiltArgs
    "spat filt params with gauss params adjusted to have anisotropy to circ_var"
    temp_filt: str
    "key to actual temp filter file name"
    max_f1_amplitude: LGNF1AmpMaxValue
    "Maximum amplitude of the F1 response after convolution"
    orientation: ArcLength[scalar]
    "orientation for the spatial filter when made anisotropic"
    circ_var: float
    "circular variance of the degree of orientation bias"
    location: RFLocation
    "location coordinates for the center of the spatial filter from the center"


@dataclass
class LGNCell(ConversionABC):
    """Single LGN Cell that provides input to a V1 Cell"""

    spat_filt: DOGSpatialFilter
    "the base spatial filter parameters to create a rendered image"
    oriented_spat_filt_params: DOGSpatFiltArgs
    "spat filt params with gauss params adjusted to have anisotropy to circ_var"
    temp_filt: TQTempFilter
    "The temporal filter to convolve with the time course of the stimulus"
    max_f1_amplitude: LGNF1AmpMaxValue
    "Maximum amplitude of the F1 response after convolution"
    orientation: ArcLength[scalar]
    "orientation for the spatial filter when made anisotropic"
    circ_var: float
    "circular variance of the degree of orientation bias"
    location: RFLocation
    "location coordinates for the center of the spatial filter from the center"

    def mk_oriented_spat_filt(self) -> DOGSpatialFilter:
        "New SpatFilter object with oriented params in place of ordinary params"

        new_sf = DOGSpatialFilter(
            parameters=self.oriented_spat_filt_params,
            optimisation_result=self.spat_filt.optimisation_result,
            ori_bias_params=self.spat_filt.ori_bias_params,
            source_data=self.spat_filt.source_data
            )

        return new_sf


    # need to rotate to orientation and get spatial_filter at specified circ_var
    # best (?):
        # rotation is a filter_function (as acts on actual array)
        # creating final array for RF is a filter function
            # takes LGNCell
            # adjusts sds for circular variance
            # makes RF array
            # rotates according to orientation

    # Need to get response to stimulus
        # use location and extent to take slice out of actual stimulus
        # convolve with this slice
        # ensure stimulus generated with appropriate extent
            # maybe just make huge so no need to worry!

# ## Full LGN Layer


@dataclass
class LGNOrientationParams(ConversionABC):
    "Distribution for orientations of LGN cells"
    mean_orientation: ArcLength[scalar]
    circ_var: float
    "variance in preferred orientation (0: high var, 1: low var)"

    @property
    def von_mises(self) -> VonMisesParams:
        return VonMisesParams.from_circ_var(
            cv=self.circ_var,
            phi=self.mean_orientation)

@dataclass
class LGNCircVarParams(ConversionABC):
    distribution_alias: str
    "attribute name of distribution defined in `orientation_preferences.circ_var_distributions`"
    circ_var_definition_method: str
    "method to be used for defining how a circular variance is converted to spat filt params"

    def __post_init__(self):

        cvvm = _cv_von_mises_module()
        # check distribution alias
        alias_available = (
            self.distribution_alias in
            AllCircVarianceDistributions.__dataclass_fields__
            )

        if not alias_available:
            raise exc.LGNError(dedent(f'''
                Circular variance distribution alias {self.distribution_alias}
                is not available.
                Options: {AllCircVarianceDistributions.__dataclass_fields__.keys()}
                ''')
                )
        # check circ_var_distribution method
        all_methods = set(CircularVarianceParams.all_methods())
        all_methods2 = set(cvvm._CircVarSDRatioMethods._all_methods())
        if all_methods.symmetric_difference(all_methods2):
            raise exc.LGNError(dedent(f'''
                circular variance method lists are inconsistent!
                data object `CircularVarianceParams` lists: {all_methods}.
                cv_von_mises lists: {all_methods2}.
                Likely issue with spatial filters and parameters for simulation!
                '''))

        if not self.circ_var_definition_method in all_methods:
            raise exc.LGNError(dedent(f'''
                circ var method ('{self.circ_var_definition_method}') not one of
                available methods ({all_methods})
                '''))


@dataclass
class LGNLocationParams(ConversionABC):
    ratio: float
    "desired ratio between sigma_x and sigma_y"
    distribution_alias: str
    "key used in rfloc_dist_index file"
    rotate_locations: bool = True
    "Whether to rotate rf locations to `orientation` angle"
    orientation: ArcLength[scalar] = ArcLength(90,'deg')
    "Rotation angle to be applied to locations for HW orientation to be non-vertical"
    # distribution_file_name: Optional[str] = None
    # "direct file name ... if preferred"

# +
@dataclass
class LGNRFLocations:
    """Locations for all cells of an LGN layer"""
    locations: Tuple[RFLocation]

    @property
    def arclength_unit(self) -> str:

        try:
            units: Set[str] = set((
                    l.arclength_unit
                    for l in self.locations
                ))
        except exc.CoordsValueError as e:
            raise exc.CoordsValueError(
                f'At least one rf location coords have inconsistent units') from e

        # not all the same unit
        if not (len(units) == 1):
            raise exc.CoordsValueError(
                f'Locations have different units: {units}')

        return units.pop()

    def array_of_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        # 2 col array (x,y)
        x_coords = np.zeros(shape=(len(self.locations)))
        y_coords = np.zeros(shape=(len(self.locations)))
        for i, loc in enumerate(self.locations):
            x_coords[i] = loc.x.value
            y_coords[i] = loc.y.value

        return x_coords, y_coords

    @classmethod
    def from_array_of_coords(
            cls, x_coords: np.ndarray, y_coords: np.ndarray,
            arclength_unit: str) -> LGNRFLocations:

        # check:              2 cols           2 dimensions
        if not (x_coords.size == y_coords.size):
            raise exc.LGNError(
                f'coords arrays are wrong shape: ({x_coords.size, y_coords.size})')

        locations = tuple(
                RFLocation(
                    x = ArcLength(x_coords[i], arclength_unit),
                    y = ArcLength(y_coords[i], arclength_unit)
                    )
                for i in range(x_coords.shape[0])
            )

        return cls(locations = locations)
# -


@dataclass
class LGNFilterParams(ConversionABC):
    spat_filters: Union[Literal['all'], Sequence[str]]
    "pick randomly from set or take 'all' if specified"
    temp_filters: Union[Literal['all'], Sequence[str]]
    "pick randomly from set or take 'all' if specified"

@dataclass
class LGNParams(ConversionABC):
    n_cells: int
    "number of cells for this LGN layer"
    orientation: LGNOrientationParams
    circ_var: LGNCircVarParams
    spread: LGNLocationParams
    filters: LGNFilterParams
    F1_amps: LGNF1AmpDistParams
    """Distribution to draw max F1 amplitudes from.

    Happily rely on default values, as these have been optimised to already
    match data from a distribution
    """
    contrast_params: Optional[ContrastParams] = None
    "Params for contrast curve to use for contrast scaling"


@dataclass
class LGNLayerRecord(ConversionABC):
    """Replace cells with cellrecords for smaller disk storage"""
    cells: Tuple[LGNCellRecord, ...]
    params: LGNParams
    rf_distance_scale: Union[None, float, ArcLength[scalar]] = None
    "if a float, represents the coefficient used to scale another metric"


@dataclass
class LGNLayer(ConversionABC):
    cells: Tuple[LGNCell, ...]
    params: LGNParams
    rf_distance_scale: Union[None, float, ArcLength[scalar]] = None
    "if a float, represents the coefficient used to scale another metric"


ContrastLgnLayerCollection = Dict[ContrastValue, Tuple[LGNLayer, ...]]
"For storing multiple lgn layers for running a simulation, by contrast of a stimulus (which affects max firing rates)"

ContrastLgnLayerCollectionRecord = Dict[ContrastValue, Tuple[LGNLayerRecord]]
"For storing on disk"

@dataclass
class LGNLayerResponse(ConversionABC):
    cell_rates: Optional[Tuple[np.ndarray, ...]]
    cell_spike_times: Tuple[Time[np.ndarray], ...]

# methods and parameters for the generation of LGN cells

# # Stimuli and Coords

@dataclass(frozen=True)
class SpaceTimeParams(ConversionABC):
    """Define size and resolution of the Spatial and Temporal Canvas

    Args:
        spat_ext: Will be the Diameter of the image canvas
            Best to ensure that this is large enough to cover the locations and sizes of
            spatial filters.

            `11 deg` or `660 mnt` works well at the moment (for standard RF location dist),
            as the required for a worst case scenario is usally around 635 mnt.
            See `stimulus.estimate_max_stimulus_spatial_ext_for_lgn()`.

        spat_res: Must have an integer value, otherwise post_init raises an error
            on instantiation
        temp_ext:
        temp_res:

    """
    spat_ext: ArcLength[scalar]
    spat_res: ArcLength[int]
    """Must have int value.
    If not, [`post_init`][utils.data_objects.SpaceTimeParams.__post_init__]
    will raise TypeError on instantiation.
    """
    temp_ext: Time[float]
    temp_res: Time[float]
    array_dtype: Optional[str] = None
    "data type for numpy arrays of coords ('float64'(default) or 'float32')"

    # manually ensure that spat_res is an integer
    # as the typing system behind val_gen doesn't allow specifying an int
    # from a float.
    def __post_init__(self):
        """Ensures spat_res has value that is an int.  Raises TypeError if not.

        This is necessary as the typing system imployed for generics doesn't
        allow for `ArcLength[int]` to not be satisfied by a `float`.
        Rather, `int` is used here for the user only.
        """
        if not isinstance(self.spat_res.value, int):
            raise TypeError(
                f'''spat_res value must be integer, instead {self.spat_res.value}
                is an {type(self.spat_res.value)}''')


@dataclass(frozen=True)
class GratingStimulusParams(ConversionABC):
    """Definition of grating stimulus"""

    spat_freq: SpatFrequency[float]
    temp_freq: TempFrequency[float]
    orientation: ArcLength[float]
    amplitude: float = 1
    DC: float = 1
    contrast: ContrastValue = ContrastValue(0.3)
    "An artifical parameter not represented in actual stimulus but simulated through correction"

    @property
    def direction(self) -> ArcLength[float]:
        """Direction of grating modulation (orientation-90 degs)"""

        direction = ArcLength(self.orientation.deg - 90, 'deg')

        return direction

    @property
    def spat_freq_x(self) -> SpatFrequency[float]:
        """Frequency in Cartesian direction derived from grating direction (ori-90)"""
        x_mag: float
        x_mag = np.cos(self.direction.rad)  # type: ignore
        freq = SpatFrequency(
            self.spat_freq.cpd * x_mag, 'cpd'  # type: ignore
            )

        return freq

    @property
    def spat_freq_y(self) -> SpatFrequency[float]:
        """Frequency in Cartesian direction derived from grating direction (ori-90)"""
        y_mag: float
        y_mag = np.sin(self.direction.rad)  # type: ignore
        freq = SpatFrequency(
            self.spat_freq.cpd * y_mag, 'cpd'  # type: ignore
            )

        return freq

# # Convolution and adjustment params

@dataclass
class EstSpatTempConvResp(ConversionABC):
    """Response amp and DC estimated from Fourier transforms of filters"""

    amplitude: float
    DC: float


@dataclass
class JointSpatTempResp(ConversionABC):
    """Response of combining a Spat and a Temp Filter"""

    amplitude: float
    "F1 Amplitude"
    DC: float
    "Shift from 0 of mean of sinusoid"
    spat_filt_DC: float
    "Original DC of spatial filter"
    temp_filt_DC: float
    "Original DC of temporal filter"


@dataclass
class ConvRespAdjParams(ConversionABC):
    """Parameters for adjusting the response of convolution"""

    amplitude: float
    "factor by which amplitude needs to be adjusted"
    DC: float
    "amount by which DC response needs to be shifted"
    joint_response: JointSpatTempResp
    max_f1_adj_factor: Optional[float] = None
    "Necessary only if scaling maximal firing rates"

@dataclass
class ConvolutionResponse(ConversionABC):
    response: np.ndarray
    adjustment_params: ConvRespAdjParams


@dataclass
class SynchronyParams(ConversionABC):
    lgn_has_synchrony: bool
    jitter: Time[float]


# # V1

@dataclass
class LIFParams(ConversionABC):
    """Leaky integrate and fire model parameters

    Default values align with those of Stanley et al 2012 and are a decent starting point
    """
    v_rest: float = -70
    "mV"
    tau_m: float = 10
    "ms"
    v_thres: float = -55
    "mV"
    v_reset: float = -65
    "mV"
    total_EPSC: float = 2.5
    "nA, nA x n_inputs (50 was Stanley default)"
    tau_EPSC: float = 0.85
    "ms"
    g_EPSC: float = 14.2
    "nsiemen"

    def get_EPSC(self, n_inputs: int) -> float:

        return self.total_EPSC / n_inputs

    def mk_dict_with_units(
            self,
            n_inputs: Union[int, Literal[False]]
            ):

        brian_units = {
            'v_rest': bnun.mV,
            'tau_m': bnun.msecond,
            'v_thres': bnun.mV,
            'v_reset': bnun.mV,
            'EPSC': bnun.nA,
            'tau_EPSC': bnun.msecond,
            'g_EPSC': bnun.nsiemens,
        }

        values = {}

        # don't normalise if False
        if n_inputs is False:
            n_inputs = 1

        for key, value in self.asdict_().items():
            # handle total_EPSC ... need to create EPSC variable manually
            if key == 'total_EPSC':
                values['EPSC'] = self.get_EPSC(n_inputs) * brian_units['EPSC']
            else:
                values[key] = value * brian_units[key]


        return values

@dataclass
class LIFNetwork(ConversionABC):
    network: bn.Network
    "Master network object for managing whole simulation"
    input_spike_generator: bn.SpikeGeneratorGroup
    "LGN Spikes providing input the V1 cell"
    spike_monitor: bn.SpikeMonitor
    "Spikes"
    membrane_monitor: bn.StateMonitor
    "Membrane potential"
    current_monitor: Optional[bn.StateMonitor] = None
    "Current, optional as unlikely to be necessary"
    initial_state_name: str = 'initial'
    "Name given to the initial state within Brian's state storage system"
    n_trials: Optional[int] = None
    "Number of trials being simulated simultaneously"

    def reset_spikes(
            self,
            spike_idxs: np.ndarray, spike_times: Time[np.ndarray]):

        self.network.restore(self.initial_state_name)
        self.input_spike_generator.set_spikes(
            indices=spike_idxs,
            times=spike_times.ms * bnun.msecond
            )

    def run(self, space_time_params: SpaceTimeParams):
        self.network.run(space_time_params.temp_ext.s * bnun.second)


@dataclass
class LIFMultiNetwork(ConversionABC):
    network: bn.Network
    "Master network object for managing whole simulation"
    input_spike_generator: bn.SpikeGeneratorGroup
    "LGN Spikes providing input the V1 cell"
    spike_monitor: bn.SpikeMonitor
    "Spikes"
    membrane_monitor: bn.StateMonitor
    "Membrane potential"
    n_simulations: int
    "Number of separate simulations run in parallel"
    n_trials: int
    "Number of trials being simulated simultaneously"
    initial_state_name: str = 'initial'
    "Name given to the initial state within Brian's state storage system"
    current_monitor: Optional[bn.StateMonitor] = None
    "Current, optional as unlikely to be necessary"

    def reset_spikes(
            self,
            spike_idxs: np.ndarray, spike_times: Time[np.ndarray],
            spikes_sorted: bool = False):

        self.network.restore(self.initial_state_name)
        self.input_spike_generator.set_spikes(
            indices=spike_idxs,
            times=spike_times.ms * bnun.msecond,
            sorted=spikes_sorted
            )

    def run(self, space_time_params: SpaceTimeParams):
        self.network.run(space_time_params.temp_ext.s * bnun.second)


@dataclass
class V1Params(ConversionABC):
    lif_params: LIFParams


# # Simulation and Results

@dataclass
class MultiStimulusGeneratorParams(ConversionABC):
    spat_freqs: Sequence[float]
    temp_freqs: Sequence[float]
    orientations: Sequence[float]
    spat_freq_unit: str = 'cpd'
    temp_freq_unit: str = 'hz'
    ori_arc_unit: str = 'deg'
    # mutable default value requires this incantation
    contrasts: Optional[Sequence[float]] = None
    amplitudes: Optional[Sequence[float]] = None
    DC_vals: Optional[Sequence[float]] = None

MultiStimulusParams = Tuple[GratingStimulusParams]

@dataclass
class SimulationParams(ConversionABC):
    n_simulations: int
    "Number of repeat simulations to run with same parameters but newly seeded random vars"
    space_time_params: SpaceTimeParams
    multi_stim_params: MultiStimulusGeneratorParams
    lgn_params: LGNParams
    lif_params: LIFParams
    n_trials: int = 10
    "Number of trials to repeat for a single simulation run with a single draw of random vars"
    analytical_convolution: bool = False
    "Whether to skip convolution and determine LGN RF responses analytically (faster)"


@dataclass
class SimulationResult(ConversionABC):
    stimulus_results_key: Optional[str]
    n_simulation: int
    spikes: Union[np.ndarray, Tuple[np.ndarray, ...]]
    lgn_responses: Union[LGNLayerResponse, Tuple[LGNLayerResponse, ...]]
    n_trials: int = 1
    membrane_potential: Optional[np.ndarray] = None

    def check_n_trials_consistency(self) -> bool:
        n = self.n_trials

        if not all((
                # if not a tuple, then only one trial
                n == (len(self.spikes) if isinstance(self.spikes, tuple) else 1),
                (
                    n == (self.membrane_potential.shape[0])
                        if self.membrane_potential is not None
                        else True
                        ),
                n == (len(self.lgn_responses) if isinstance(self.lgn_responses, tuple) else 1),
                )):
            return False
        else:
            return True

    def get_spikes(self, n_trial: int) -> np.ndarray:
        if isinstance(self.spikes, tuple):
            return self.spikes[n_trial]
        else:
            return self.spikes

    def get_mem_pot(self, n_trial: int) -> Optional[np.ndarray]:
        value = (
            self.membrane_potential[n_trial, :]
                if self.membrane_potential is not None
                else None
                )
        return value

    def get_lgn_response(self, n_trial: int) -> LGNLayerResponse:
        if isinstance(self.lgn_responses, tuple):
            return self.lgn_responses[n_trial]
        else:
            return self.lgn_responses


@dataclass
class SimulationMetaData():
    exp_id: str
    comments: str

    @property
    def creation_time_str(self) -> str:
        return self.creation_time.isoformat()

    def __post_init__(self):
        self.creation_time = dt.datetime.utcnow()


@dataclass
class SimulationResults(ConversionABC):
    params: SimulationParams
    lgn_layers: ContrastLgnLayerCollection
    results: Dict[str, Tuple[SimulationResult, ...]]
