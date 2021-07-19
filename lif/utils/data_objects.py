"""Classes for handling and grouping basic data objects"""

from __future__ import annotations
from functools import partial
from typing import Union, Optional, Iterable, Dict, Any, Tuple, List
from dataclasses import dataclass, astuple, asdict, field
import datetime as dt
from pathlib import Path
import pickle as pkl
import copy

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.interpolate import interp1d

from . import settings
from .units.units import (
    ArcLength, TempFrequency, SpatFrequency, Time, val_gen
    )
from ..receptive_field.filters import filter_functions as ff


# numerical_iter = Union[np.ndarray, Iterable[float]]

PI: float = np.pi  # type: ignore


class ConversionABC:
    "ABC for adding datclasses.asdict and .astuple as methods"

    def asdict_(self) -> Dict[str, Any]:
        return asdict(self)

    def astuple_(self) -> Tuple[Any, ...]:
        return astuple(self)


@dataclass
class CitationMetaData(ConversionABC):
    "Author etc for source of data for a temp_filt fit"
    author: str
    year: int
    title: str
    reference: str
    doi: Optional[str] = None

    def _set_dt_uid(self, reset: bool = False) -> None:
        "Set a datetime based unique ID"

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


# > Temp Filter

@dataclass
class TFRespMetaData(ConversionABC):
    "Metadata or additional data on context of temp filter data"
    dc: float
    sf: SpatFrequency[float]
    mean_lum: float
    contrast: Optional[float] = None


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
    "All data and information on temporal frequency response from literature"
    data: TempFiltData
    resp_params: TFRespMetaData
    meta_data: Optional[CitationMetaData] = None


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
        """[tau.value, w, phi]
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
        """Create object from iterable: (a, tau, w, phi)

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


@dataclass
class TQTempFilter(ConversionABC):
    "Full tq temp filter that has been fit to data from the literature"

    source_data: TempFiltParams
    parameters: TQTempFiltParams
    optimisation_result: OptimizeResult

    def save(self, overwrite: bool = False):

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
        "Return list of temporal filters saved in data directory"

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

        with open(path, 'rb') as f:
            temp_filt = pkl.load(f)

        return temp_filt


# > Spatial Filter

@dataclass
class SpatFiltData(ConversionABC):

    amplitudes: np.ndarray
    frequencies: SpatFrequency[np.ndarray]


@dataclass
class SFRespMetaData(ConversionABC):
    "Metadata or additional data on context of spat filter data"
    dc: float
    tf: TempFrequency[float]  # temp frequency
    mean_lum: float
    contrast: Optional[float] = None


@dataclass
class SpatFiltParams(ConversionABC):
    "All data and information on spatial frequency response from the literature"
    data: SpatFiltData
    resp_params: SFRespMetaData
    meta_data: Optional[CitationMetaData] = None


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


@dataclass
class Gauss2DSpatFiltParams(ConversionABC):
    """Args for a 2d gaussian spatial filter

    amplitude: magnitude
    arguments: Gauss2DSpatFiltArgs (main shape arguments)

    """
    amplitude: float
    arguments: Gauss2DSpatFiltArgs

    def array(self) -> np.ndarray[np.float64]:
        return np.r_[self.amplitude, self.arguments.array()]  # type: ignore

    @classmethod
    def from_iter(cls, data: Iterable[float], arclength_unit: str = 'deg') -> Gauss2DSpatFiltParams:
        """Create object from iterable: (a, h_sd, v_sd)
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
        """Create object from iterable: (a, h_sd, v_sd)
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
        "cent args, surr args"
        return np.hstack((self.cent.array(), self.surr.array()))

    @classmethod
    def from_iter(cls, data: Iterable[float], arclength_unit: str = 'deg') -> DOGSpatFiltArgs1D:
        "cent_a, cent_sd, surr_a, surr_sd"
        cent_a, cent_sd, surr_a, surr_sd = data

        cent = Gauss1DSpatFiltParams(cent_a, ArcLength(cent_sd, arclength_unit))
        surr = Gauss1DSpatFiltParams(surr_a, ArcLength(surr_sd, arclength_unit))

        return cls(cent=cent, surr=surr)


@dataclass
class DOGSpatFiltArgs(ConversionABC):
    """Args for a DoG spatial filter: 2 2dGauss, one subtracted from the other
    """
    cent: Gauss2DSpatFiltParams
    surr: Gauss2DSpatFiltParams

    def array(self) -> np.ndarray:
        "1D array of [cent args, surr args]"
        return np.hstack((self.cent.array(), self.surr.array()))

    def array2(self) -> np.ndarray:
        "2D array of [[cent args], [surr args]]"
        return np.vstack((self.cent.array(), self.surr.array()))

    def to_dog_1d(self) -> DOGSpatFiltArgs1D:
        "Take h_sd values, presume radial symmetry, return 1D DoG spat fil"

        cent_a, cent_h_sd, _, surr_a, surr_h_sd, _ = self.array()
        return DOGSpatFiltArgs1D.from_iter([cent_a, cent_h_sd, surr_a, surr_h_sd])

    def mk_ori_biased_duplicate(
            self, v_sd_factor: float, h_sd_factor: float
            ) -> DOGSpatFiltArgs:
        "Create duplicate but with factors applies to sd values"

        spat_filt_args = copy.deepcopy(self)

        spat_filt_args.cent.arguments.v_sd = ArcLength(
            spat_filt_args.cent.arguments.v_sd.mnt * v_sd_factor, 'mnt'
            )
        spat_filt_args.cent.arguments.h_sd = ArcLength(
            spat_filt_args.cent.arguments.h_sd.mnt * h_sd_factor, 'mnt'
            )

        return spat_filt_args


@dataclass
class CircularVarianceSDRatioVals(ConversionABC):
    "Circular Variance to SD ration values"

    sd_ratio_vals: np.ndarray
    circular_variance_vals: np.ndarray

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
        if not np.all(circ_var >= 0) and np.all(circ_var <= 1):  # type: ignore
            raise ValueError(f'Circ var must be between 0 and 1, is {circ_var}')

        return self._sd_ratio(circ_var)

    def _mk_interpolated(self):
        "Add methods for interpolation in both directions (using interp1d)"

        interp_func = partial(interp1d, fill_value='extrapolate')

        self._sd_ratio = interp_func(
            self.circular_variance_vals, self.sd_ratio_vals)
        self._circ_var = interp_func(
            self.sd_ratio_vals, self.circular_variance_vals)


@dataclass
class DOGSpatialFilter(ConversionABC):

    "Full DoG Spatial filter that has been fit to data from the literature"

    source_data: SpatFiltParams
    parameters: DOGSpatFiltArgs
    optimisation_result: OptimizeResult
    ori_bias_params: CircularVarianceSDRatioVals

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
        "Return list of spatial filters saved in data directory"

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

        with open(path, 'rb') as f:
            spat_filt = pkl.load(f)

        return spat_filt

    def mk_ori_biased_parameters(
            self, circ_var: float) -> DOGSpatFiltArgs:
        """Copied parameters with sds adjusted to produce circular variance"""

        sd_ratio = self.ori_bias_params.circ_var2ratio(circ_var)
        v_sd_fact, h_sd_fact = ff.mk_ori_biased_sd_factors(sd_ratio)

        new_sf_params = self.parameters.mk_ori_biased_duplicate(
            v_sd_fact, h_sd_fact
            )

        return new_sf_params


# > Full LGN Model

@dataclass
class LGNCell(ConversionABC):
    "Single LGN Cell that provides input to a V1 Cell"

    spat_filt: DOGSpatialFilter
    temp_filt: TQTempFilter
    orientation: ArcLength[float]
    circ_var: float
    location: Tuple[float, float]  # tentative, maybe requires own data object

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


# > Stimuli and Coords

@dataclass
class SpaceTimeParams(ConversionABC):
    "Spatial and Temporal Canvas"

    spat_ext: ArcLength[float]
    spat_res: ArcLength[float]
    temp_ext: Time[float]
    temp_res: Time[float]


@dataclass
class GratingStimulusParams(ConversionABC):
    "Definition of grating stimulus"

    spat_freq: SpatFrequency[float]
    temp_freq: TempFrequency[float]
    orientation: ArcLength[float]
    amplitude: float = 1
    DC: float = 1

    @property
    def direction(self) -> ArcLength[float]:
        "Direction of grating modulation (orientation-90 degs)"

        direction = ArcLength(self.orientation.deg - 90, 'deg')

        return direction

    @property
    def spat_freq_x(self) -> SpatFrequency[float]:
        "Frequency in cartesion direction derived from grating direction (ori-90)"
        x_mag: float
        x_mag = np.cos(self.direction.rad)  # type: ignore
        freq = SpatFrequency(
            self.spat_freq.cpd * x_mag, 'cpd'  # type: ignore
            )

        return freq

    @property
    def spat_freq_y(self) -> SpatFrequency[float]:
        "Frequency in cartesion direction derived from grating direction (ori-90)"
        y_mag: float
        y_mag = np.sin(self.direction.rad)  # type: ignore
        freq = SpatFrequency(
            self.spat_freq.cpd * y_mag, 'cpd'  # type: ignore
            )

        return freq



@dataclass
class EstSpatTempConvResp(ConversionABC):
    "Response amp and dc estimated from fourier transfroms of filters"

    amplitude: float
    DC: float


@dataclass
class JointSpatTempResp(ConversionABC):
    "Response of combining a Spat and a Temp Filt"

    ampitude: float
    DC: float


@dataclass
class ConvRespAdjParams(ConversionABC):
    "Parameters for adjusting the response of convolution"

    amplitude: float
    DC: float
